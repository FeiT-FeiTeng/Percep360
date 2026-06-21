import os
import math
from inspect import isfunction

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch.utils import checkpoint
import torchvision
from torchvision.utils import make_grid

from einops import rearrange, repeat




def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
    
class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, heads=8, dim_head=64, dropout=0, atten_map_res=[32, 48], max_boxes=80, max_length=77):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads


        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)

        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm2 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)

        self.to_out = nn.Sequential( nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )


        self.atten_map_res = atten_map_res
        self.max_boxes = max_boxes
        self.max_length = max_length

        self.threshold = -0.001

        self.lamda1 = 5.0
        self.lamda2 = 5.0      


    def fill_inf_from_mask(self, sim, mask):
        if mask is not None:
            B,M = mask.shape
            mask = mask.unsqueeze(1).repeat(1,self.heads,1).reshape(B*self.heads,1,-1)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)
        return sim 


    def forward(self, x, key, value, mask=None, perl_box_masking_map=None, perl_road_masking_map=None):

        q = self.norm1(self.to_q(x))     # torch.Size([1, 840, 320])

        k_attn = F.softmax(key, dim=1)  # Random weights; can be replaced by learnable weights.
        k_out = (k_attn).sum(dim=1, keepdim=True) 
        k = self.norm2(self.to_k(k_out))   
        # torch.Size([1, 840, 320])  torch.Size([1, 840, 1])
        v_attn = F.softmax(value, dim=1)  # Random weights; can be replaced by learnable weights.
        v_out = (v_attn).sum(dim=1, keepdim=True) 
        v = self.norm3(self.to_v(v_out)) # torch.Size([1, 840, 320])  torch.Size([1, 840, 1])
   
        B, N, HC = q.shape  # 1 840 320
        _, M, _ = k.shape # 840
        H = self.heads
        C = HC // H  # 40 

        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C  torch.Size([8, 840, 40])
        k = k.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) # (B*H)*M*C  torch.Size([8, 840, 40])
        v = v.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) # (B*H)*M*C  torch.Size([8, 840, 40])

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale # torch.Size([1, 840, 840])
        self.fill_inf_from_mask(sim, mask) #torch.Size([8, 840, 840])

        if perl_box_masking_map != None:

            perl_box_masking_map = repeat(perl_box_masking_map, 'b m n->(b h) n m', h=H) # (B*H)*N*M

            sim1 = sim + self.lamda1 * perl_box_masking_map # (B*H)*N*M

        if perl_road_masking_map != None:
            #print("Mask embedding")
            perl_road_masking_map = repeat(perl_road_masking_map, "b m n->(b h) n m", h=H,m=1) # B*1*N -> (B*H)*N*M

            sim2 = sim + self.lamda2 * perl_road_masking_map # (B*H)*N*M

            sim = torch.maximum(sim1, sim2)
        
        attn = sim.softmax(dim=-1) # (B*H)*N*M

        out = torch.einsum('b i j, b j d -> b i d', attn, v) # (B*H)*N*C
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B*N*(H*C)

        return self.to_out(out)
    


class GatedCrossAttentionDense(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head):
        super().__init__()
        
        self.attn = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads, dim_head=d_head) 
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  

    def forward(self, x, objs, perl_box_masking_map=None, perl_road_masking_map=None):  #x=x,objs=road_map_embedding,perl_road_masking_map=perl_road_masking_map
        # x = self.attn_back(x, road_map_embedding, perl_road_masking_map=perl_road_masking_map) + x
        buffer = x
        x = x + self.scale*torch.tanh(self.alpha_attn) * self.attn( self.norm1(x), objs, objs, 
                                                                   perl_box_masking_map=perl_box_masking_map, 
                                                                   perl_road_masking_map=perl_road_masking_map) 
        x = buffer + self.scale*torch.tanh(self.alpha_dense) * self.ff( self.norm2(x) ) 
        
        return x 
    


class BGasicTransformerBlock(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head, fuser_type, use_checkpoint=True, num_camera=6):
        super().__init__()
        self.num_camera= num_camera 
        self.ff = FeedForward(query_dim, glu=True)
        self.attn2 = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads, dim_head=d_head) 

        self.cross_view_left = CrossAttention(query_dim=query_dim, key_dim=query_dim, value_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.cross_view_right = CrossAttention(query_dim=query_dim, key_dim=query_dim, value_dim=query_dim, heads=n_heads, dim_head=d_head)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)

        self.use_checkpoint = use_checkpoint

        if fuser_type == "gatedCA":
            self.fuser = GatedCrossAttentionDense(query_dim, key_dim, value_dim, n_heads, d_head) 
            self.attn_back = GatedCrossAttentionDense(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, n_heads=n_heads, d_head=d_head)
        else:
            assert False, f"fuser_type={fuser_type} is not supported!!!"


    def forward(self, x, context, objs, perl_box_masking_map, perl_road_masking_map, road_map_embedding):
        if self.use_checkpoint and x.requires_grad:
            return checkpoint.checkpoint(self._forward, x, context, objs, perl_box_masking_map, perl_road_masking_map, road_map_embedding)
        else:
            return self._forward(x, context, objs, perl_box_masking_map, perl_road_masking_map, road_map_embedding)

    def _forward(self, x, context, objs, perl_box_masking_map, perl_road_masking_map, road_map_embedding): 
        buffer = x
        if perl_box_masking_map is not None:
            if perl_road_masking_map.shape[2] != x.shape[1]:
                perl_road_masking_map = F.interpolate(
                    perl_road_masking_map,  # [1, 1, 1, 840] with channel dimension.
                    size=x.shape[1],                    # Target length.
                    mode='linear'                        # Linear interpolation.
                )  
            if perl_box_masking_map.shape[2] != x.shape[1]:
                perl_box_masking_map = F.interpolate(
                    perl_box_masking_map,  # [1, 1, 1, 840] with channel dimension.
                    size=x.shape[1],                    # Target length.
                    mode='linear'                        # Linear interpolation.
                ) 
            x = self.fuser(x, context, perl_box_masking_map=perl_box_masking_map ,perl_road_masking_map=perl_road_masking_map)

        
        x = self.attn2(self.norm2(x), context, context) + x

        x = self.ff(self.norm3(x)) + buffer

        return x


if __name__=="__main__":
    import torch
    x = torch.rand(50,840,320)
    context = torch.rand(50,840,320)
    perl_box_masking_map = torch.rand(50,1,840)
    perl_road_masking_map = torch.rand(50,1,840)
    query_dim = 320
    key_dim = 320
    value_dim = 320
    n_heads = 8
    fuser_type ="gatedCA"
    d_head = query_dim//n_heads

    # flattened_m = rearrange(perl_box_masking_map, "b c h w -> b c (h w)")
    # perl_box_masking_map = F.interpolate(flattened_m, size=840, mode="linear")

    # flattened_r = rearrange(perl_road_masking_map, "b c h w -> b c (h w)")
    # perl_road_masking_map = F.interpolate(flattened_r, size=840, mode="linear")

    model = BGasicTransformerBlock(query_dim, key_dim, value_dim, n_heads, d_head, fuser_type, use_checkpoint=True, num_camera=1)
    out = model(x=x,context=context,objs=None,perl_box_masking_map=perl_box_masking_map,perl_road_masking_map=perl_road_masking_map,road_map_embedding=None)

    print(out.shape)
