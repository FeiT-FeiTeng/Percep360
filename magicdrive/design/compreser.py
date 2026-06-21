import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import torchvision.transforms as transforms
#from deform2d import deform_conv2d
class DeformableProjEmbed(nn.Module):  # Deformable Patch Embedding
    """feature map to Projected Embedding
    """
    def __init__(self, in_chans=512, emb_chans=128, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.proj = nn.Conv2d(in_chans, emb_chans, kernel_size=kernel_size, stride=stride, padding=padding,padding_mode="circular")
        # --- deformable offset and modulator
        self.offset_conv = nn.Conv2d(in_chans, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding,padding_mode="circular")

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        self.modulator_conv = nn.Conv2d(in_chans, 1 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding,padding_mode="circular")
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        self.norm = nn.BatchNorm2d(emb_chans)
        self.act = nn.GELU()

    def deform_proj(self, x):
        max_offset = min(x.shape[-2], x.shape[-1]) // 4
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.proj.weight,
                                          bias=self.proj.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride)  # deformable conv
        return x

    def forward(self, x):
        x = self.deform_proj(x)
        x = self.act(self.norm(x))
        return x

class FUadapter(nn.Module):
    def __init__(self, dim_in=4, dim_out=4):
        super(FUadapter, self).__init__()
        self.dim_in = dim_in
        self.emb_chans = 16
        self.DP = DeformableProjEmbed(in_chans=self.dim_in,emb_chans=self.emb_chans)
        dim_query=1
        
        self.norm = nn.GELU()
        
        self.spatial_query_generator = nn.Conv2d(in_channels=2, out_channels=dim_query, kernel_size=1)
        self.channel_query_generator = nn.Linear(self.emb_chans * 2, self.emb_chans)  
        
        self.down = DeformableProjEmbed(in_chans=self.emb_chans*2,emb_chans=1) 
        self.down2 = DeformableProjEmbed(self.emb_chans+1,emb_chans=dim_in)        

    def forward(self, x,y): # x Img, y Line
        buffer = x  # Keep the residual connection.
        B,C,H,W = x.shape
        x = self.DP(x)
        y = self.DP(y)
        
        Bb,Cc,Hh,Ww = x.shape
        
        SP_mean_pool = torch.mean(y, dim=1).unsqueeze(1)  #torch.Size([1, 1, 20, 300])
        SP_max_pool = torch.amax(y, dim=1).unsqueeze(1)	 #torch.Size([1, 1, 20, 300])
        
        Ch_mean_pool = torch.mean(y, dim=(2, 3))			#torch.Size([1, 16])
        Ch_max_pool = torch.amax(y, dim=(2, 3))				#torch.Size([1, 16])

        y_SP_query = self.spatial_query_generator(torch.cat((SP_mean_pool, SP_max_pool), dim=1))
        y_SP_query = y_SP_query.expand(Bb,Cc,Hh,Ww) + x  #torch.Size([1, 16, 20, 300])
        
        y_Ch_query = self.channel_query_generator(torch.cat((Ch_mean_pool, Ch_max_pool), dim=-1))
        y_Ch_query = y_Ch_query.unsqueeze(2).unsqueeze(3).expand(Bb,Cc,Hh,Ww) + x #torch.Size([1, 16, 20, 300])
        
        Line = torch.cat((y_SP_query, y_Ch_query), dim=1)  #torch.Size([1, 32, 20, 300])
        Line = self.down(Line) #torch.Size([1, 1, 20, 300])
        
        out = torch.cat((Line, x), dim=1) #torch.Size([1, 17, 20, 300])
        out = self.down2(out)   #torch.Size([1, 4, 20, 300])
        out = out + buffer
        return out
    
    
if __name__ =="__main__":
    import torch
    x = torch.randn(1, 4, 10, 150)
    model = FUadapter()
    out = model(x,x)
    print(out)
