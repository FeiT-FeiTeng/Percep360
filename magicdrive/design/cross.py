from einops import rearrange
import torch.nn as nn
class SegformercrosAttn(nn.Module):
    def __init__(self, dim=4, num_heads=2, sr_ratio=1, dropout=0.1, activation=nn.GELU):
        super(SegformercrosAttn, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5  # Scaling factor for QK^T
        self.sr_ratio = sr_ratio
        # Query, Key, Value Linear Layers
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        # Optional reduction in Key/Value spatial size (for efficiency)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # Activation layer
        self.act = activation()
    def forward(self, x, y):  #x - IMG, Y --Depth
        buffer = x
        B, C, H, W = x.shape  # BCHW input.
        # Flatten x and y to [B, N, C], where N = H * W.
        x = rearrange(x, "b c h w -> b (h w) c")  #torch.Size([2, 1024, 64])
        y = rearrange(y, "b c h w -> b (h w) c") 
        B, N, C = x.shape
        # Compute query
        q = self.q(x)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads) #torch.Size([2, 1024, 64])
        # Compute key and value
        if self.sr_ratio > 1:
            y_ = y.permute(0, 2, 1).reshape(B, C, H, W)
            y_ = self.sr(y_).reshape(B, C, -1).permute(0, 2, 1)  #torch.Size([2, 64, 32, 32])
            y_ = self.norm(y_)
        else:
            y_ = y
        kv = self.kv(y_)  #torch.Size([2, 256, 64])
        k, v = kv.chunk(2, dim=-1)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads) #torch.Size([2, 8, 256, 8])
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads) #torch.Size([2, 8, 256, 8])
        # Attention mechanism
        attn = (q @ k.transpose(-2, -1)) * self.scale  #torch.Size([2, 8, 1024, 256])
        attn = attn.softmax(dim=-1)
        # Apply dropout to attention scores
        attn = self.dropout(attn)
        # Aggregate and apply dropout to the value
        out = (attn @ v)     #torch.Size([2, 8, 1024, 8])
        out = self.dropout(out) #torch.Size([2, 8, 1024, 8])
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.act(out)
        # Rearrange and apply activation
        out = rearrange(out, "b (h w) c -> b c h w", h=H, w=W)
        out = out + buffer
        return out
    
if __name__ == "__main__":
    import torch
    B, C, H, W = 1, 4, 300, 20  # Test tensor size.
    x = torch.randn(1, 4, 10, 150)
    y = torch.randn(1, 4, 10, 150)

    model = SegformercrosAttn(dim=C, num_heads=2)
    output = model(x, y)
    print(output.shape)  # Expected output: [B, C, H, W].
