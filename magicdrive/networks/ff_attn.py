import torch
import torch.nn as nn
import torch.fft
from diffusers.models.attention_processor import Attention

class AmplitudeScaling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x)) * x


class PhaseContinuity(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        nn.init.constant_(self.conv.weight, 1/25)

    def forward(self, x):
        smoothed_phase = self.conv(x)
        return torch.tanh(smoothed_phase) * x

class ComplexConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_real = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv_imag = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        real = x.real
        imag = x.imag
        real_out = self.conv_real(real.half()) - self.conv_imag(imag.half())
        imag_out = self.conv_real(imag.half()) + self.conv_imag(real.half())
        return torch.complex(real_out.float(), imag_out.float())
    
class OS_SSM(nn.Module):
    def __init__(self, channels,heads,dim_head,dropout,bias,upcast_attention):
        super().__init__()
        #self.input_conv = nn.Conv2d(channels, channels, kernel_size=1)

        # Stabilization stage.
        self.amplitude_scaling = AmplitudeScaling(channels)
        self.phase_continuity = PhaseContinuity(channels)

        self.complex_conv = ComplexConv(channels)

        # Enhancement stage.
        self.model_amp = Attention(
            query_dim=channels,
            cross_attention_dim=channels,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            bias=bias,
            upcast_attention=upcast_attention,
        )


    def forward(self, x):
        #x = self.input_conv(x)
        #print("start ffp")
        # FFT
        x = x.to(torch.float32)
        fft_x = torch.fft.fft(x,dim=1) # B,N,C
        amp = torch.abs(fft_x)# B,N,C
        pha = torch.angle(fft_x)# B,N,C
        amp = amp.half()
        pha = pha.half()
        amp = amp.transpose(1, 2)# B,C,N
        pha = pha.transpose(1, 2)# B,C,N

        # Stabilize amplitude and phase.
        amp_scaled = self.amplitude_scaling(amp)# B,C,N
        pha_continuous = self.phase_continuity(pha)# B,C,N
        
        complex_input = torch.complex(amp_scaled.float(), pha_continuous.float())# B,C,N
        complex_processed = self.complex_conv(complex_input)# B,C,N

        # Extract amplitude and phase.
        processed_amp = torch.abs(complex_processed).half()# B,C,N
        processed_pha = torch.angle(complex_processed).half()# B,C,N

        # Further enhance amplitude and phase with attention.
        processed_amp = self.model_amp(processed_amp.transpose(1, 2)).transpose(1, 2)# B,C,N
        processed_pha = self.model_amp(processed_pha.transpose(1, 2)).transpose(1, 2)# B,C,N

        # Reconstruct through the complex residual path.
        combined_fft = torch.complex(processed_amp.float(), processed_pha.float()) # B,C,N
        ifft_x = torch.fft.ifft(combined_fft,dim=2).real
        ifft_x = ifft_x.half()

        # Gated Deconv FFN
        out = ifft_x.transpose(1, 2).half()

        return out

if __name__ == "__main__":
    # Example initialization.
    channels = 64
    heads = 8
    dim_head = 64  # Usually 32 or 64; adjust as needed.
    dropout = 0.1  # Recommended range: 0.1 to 0.3.
    bias = True
    upcast_attention = False  # Usually False unless attention overflows.

    model = OS_SSM(
        channels=channels,
        heads=heads,
        dim_head=dim_head,
        dropout=dropout,
        bias=bias,
        upcast_attention=upcast_attention
    ).cuda().half()  # Move to CUDA/half precision as needed.

    # Example input test.
    x = torch.randn(8, 1500,channels).cuda().half()
    output = model(x)

    print(output.shape)  # Expected shape.
