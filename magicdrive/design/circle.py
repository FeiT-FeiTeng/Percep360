
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

if __name__ == "__main__":
    import torch
    import torchvision
    print(torch.__version__)
    print(torchvision.__version__)
    #1.10.2+cu113
    #0.11.3+cu102

    # Load and preprocess the image
    img_path = "assets/example.jpg"  # Specify the correct image path here.
    image = Image.open(img_path).convert('RGB')

    # Preprocessing: resize and convert image to tensor
    preprocess = transforms.Compose([
        transforms.Resize((1900, 600)),
        transforms.ToTensor()
    ])

    input_image = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Initialize the deformable projection layer
    model = DeformableProjEmbed(in_chans=3, emb_chans=3, kernel_size=3, stride=1, padding=1)

    # Apply the deformable convolution
    output = model(input_image) 

    # Output the shape of the result
    print("Output shape:", output.shape)
