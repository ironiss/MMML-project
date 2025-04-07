import torch.nn as nn
import torch
import torch.nn.functional as F

class MAWarpBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.offset_net = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1),  
            nn.ReLU(),
            nn.Conv2d(channels, 2, 3, padding=1)            
        )
        
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels * 2, 1, 3, padding=1),       
            nn.Sigmoid()                                    
        )
        
        self.res_net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, h, phi_prev, phi_next, m_prev, m_next):      
        m_prev = F.interpolate(m_prev, size=h.shape[-2:], mode='bilinear')
        m_next = F.interpolate(m_next, size=h.shape[-2:], mode='bilinear')
        
        Omega_prev = self.offset_net(torch.cat([h, m_prev, phi_prev], dim=1))
        Omega_next = self.offset_net(torch.cat([h, m_next, phi_next], dim=1))
        
        h_prev = self.warp(phi_prev, Omega_prev)
        h_next = self.warp(phi_next, Omega_next)
        
        g = self.gate_net(torch.cat([h_prev, h_next], dim=1))
        delta = self.res_net(h)
        return g * h_prev + (1 - g) * h_next + delta


    def warp(self, x, offset):
        B, _, H, W = x.size()
        
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device)
        )
        grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).repeat(B, 1, 1, 1)
        
        grid = grid + offset.permute(0, 2, 3, 1)        
        return F.grid_sample(x, grid, padding_mode='border', align_corners=True)



class MainDecoder(nn.Module):
    def __init__(self, out_channels=3, channels=128, codebook_dim=256, num_res_blocks=2):
        super().__init__()
        
        self.conv_in = nn.Conv2d(codebook_dim, channels*4, 1)
        
        self.res_blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(channels*4, channels*4, 3, padding=1),
                nn.GroupNorm(32, channels*4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels*4, channels*4, 3, padding=1),
                nn.GroupNorm(32, channels*4)
            ) for _ in range(num_res_blocks)
        ])
        
        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(channels*4, channels*2, 4, stride=2, padding=1),
                nn.GroupNorm(32, channels*2),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(channels*2, channels, 4, stride=2, padding=1),
                nn.GroupNorm(32, channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(channels, out_channels, 4, stride=2, padding=1),
                nn.Tanh()  
            )
        ])
        

        self.ma_warps = nn.ModuleList([
            MAWarpBlock(channels*4),
            MAWarpBlock(channels*2),
            MAWarpBlock(channels)
        ])


    def forward(self, z, phi_prev, phi_next, m_prev, m_next):
        h = self.conv_in(z)
        h = h + self.res_blocks(h)
        
        for i, (up_block, ma_warp) in enumerate(zip(self.up_blocks, self.ma_warps)):
            h = up_block(h)
            
            if i < len(self.ma_warps) - 1: 
                h = ma_warp(h, 
                           F.interpolate(phi_prev, scale_factor=0.5**i, mode='bilinear'),
                           F.interpolate(phi_next, scale_factor=0.5**i, mode='bilinear'),
                           F.interpolate(m_prev, size=h.shape[-2:], mode='bilinear'),
                           F.interpolate(m_next, size=h.shape[-2:], mode='bilinear'))
        
        return h
    
