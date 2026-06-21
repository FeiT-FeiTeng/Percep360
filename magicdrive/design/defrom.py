import torch
import torch.nn as nn
import torchvision

class DeformableProjEmbed(nn.Module):  # Deformable Patch Embedding
    """feature map to Projected Embedding"""
    def __init__(self, in_chans, emb_chans, kernel_size=3, stride=2, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Projection layer
        self.proj = nn.Conv2d(in_chans, emb_chans, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode="zeros")
        
        # Deformable offset and modulator
        self.offset_conv = nn.Conv2d(in_chans, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode="zeros")
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_chans, 1 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode="zeros")
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.norm = nn.BatchNorm2d(emb_chans)
        self.act = nn.GELU()

        self.printed = True
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
        
        if self.printed:
            print("Line EMB")
            self.printed = False
        
        x = self.deform_proj(x)
        x = self.act(self.norm(x))
        return x

class MSPoolAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        pools = [3,7,11]
        self.conv0 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.pool1 = nn.AvgPool2d(pools[0], stride=1, padding=pools[0]//2, count_include_pad=False)
        self.pool2 = nn.AvgPool2d(pools[1], stride=1, padding=pools[1]//2, count_include_pad=False)
        self.pool3 = nn.AvgPool2d(pools[2], stride=1, padding=pools[2]//2, count_include_pad=False)
        self.conv4 = nn.Conv2d(dim, dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.GELU()

    def forward(self, x):
        u = x.clone()
        x_in = self.conv0(x)
        x_1 = self.pool1(x_in)
        x_2 = self.pool2(x_in)
        x_3 = self.pool3(x_in)
        x_out = self.sigmoid(self.conv4(x_in + x_1 + x_2 + x_3)) * u
        return self.act(x_out + u)

class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define each layer module.
        self.layer1_proj = DeformableProjEmbed(in_chans=3, emb_chans=8, kernel_size=3, stride=2, padding=0)
        self.layer1_attn = MSPoolAttention(8)
  
        self.layer2_proj = DeformableProjEmbed(in_chans=8, emb_chans=16, kernel_size=3, stride=2, padding=0)
        self.layer2_attn = MSPoolAttention(16)

        self.layer4_proj = DeformableProjEmbed(in_chans=16, emb_chans=4, kernel_size=3, stride=1, padding=1)
        self.layer4_attn = MSPoolAttention(4)

    def forward(self, x):
        # First layer.
        x = x.squeeze(1)
        x = self.layer1_proj(x)
        x = self.layer1_attn(x)

        # Second layer.
        x = self.layer2_proj(x)
        x = self.layer2_attn(x)

        # Resize by interpolation.
        x = nn.functional.interpolate(x, size=(10, 150), mode='bilinear')


        # Fourth layer.
        x = self.layer4_proj(x)
        x = self.layer4_attn(x)

        return x

if __name__ == "__main__":
    # Create an input tensor.
    x = torch.randn(1, 1, 3, 80, 1200)

    # Instantiate the model.
    model = FullModel()

    # Forward pass.
    output = model(x)
