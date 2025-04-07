import torch.nn as nn
from VectorQuantizer import VectorQuantizer

class MainEncoder(nn.Module):
    def __init__(self, in_channels=3, channels=128, num_res_blocks=2, codebook_size=8192, codebook_dim=256):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, channels, 3, stride=1, padding=1)
        
        self.down_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels*2, 4, stride=2, padding=1),
                nn.GroupNorm(32, channels*2),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(channels*2, channels*4, 4, stride=2, padding=1),
                nn.GroupNorm(32, channels*4),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(channels*4, channels*4, 4, stride=2, padding=1),
                nn.GroupNorm(32, channels*4),
                nn.ReLU(inplace=True)
            )
        ])
        
        self.res_blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(channels*4, channels*4, 3, padding=1),
                nn.GroupNorm(32, channels*4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels*4, channels*4, 3, padding=1),
                nn.GroupNorm(32, channels*4)
            ) for _ in range(num_res_blocks)
        ])
        
        self.conv_out = nn.Conv2d(channels*4, codebook_dim, 1)
        
        self.quantizer = VectorQuantizer(codebook_size, codebook_dim)
        
    def forward(self, x):
        h = self.conv_in(x)
        
        for block in self.down_blocks:
            h = block(h)
        
        h = h + self.res_blocks(h)
        h = self.conv_out(h)
        
        quantized, vq_loss = self.quantizer(h)
        return quantized, vq_loss, h