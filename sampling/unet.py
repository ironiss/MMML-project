
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.same_channels = in_channels == out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if not self.same_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.bn_shortcut = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if not self.same_channels:
            identity = self.bn_shortcut(self.conv_shortcut(identity))
        
        out += identity
        out = F.relu(out)
        
        return out

class MaxViTBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4):
        super(MaxViTBlock, self).__init__()
        self.dim = dim
        self.heads = heads
        
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.bn2 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(dim)
        
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        residual = x
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.depthwise_conv(x)))
        x = self.bn3(self.conv2(x))
        x = x + residual
        
        residual = x
        x_spatial = x.permute(0, 2, 3, 1)  # B, H, W, C
        x_spatial = x_spatial.reshape(B, H * W, C)
        x_spatial = self.spatial_norm(x_spatial)
        x_spatial, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        x_spatial = x_spatial.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + x_spatial
        
        residual = x
        x_channel = x.reshape(B, C, H * W).permute(0, 2, 1)  # B, H, W, C
        x_channel = self.channel_norm(x_channel)
        x_channel, _ = self.channel_attn(x_channel, x_channel, x_channel)
        x_channel = x_channel.permute(0, 2, 1).reshape(B, C, H, W)
        x = x + x_channel
        
        residual = x
        x_mlp = x.permute(0, 2, 3, 1)  # B, H, W, C
        x_mlp = x_mlp.reshape(B, H * W, C)
        x_mlp = self.mlp_norm(x_mlp)
        x_mlp = self.mlp(x_mlp)
        x_mlp = x_mlp.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + x_mlp
        
        return x

class MaxCrossAttentionBlock(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, heads=8):
        super(MaxCrossAttentionBlock, self).__init__()
        self.output_dim = decoder_dim
        
        self.proj_dim = max(encoder_dim, decoder_dim)
        self.encoder_proj = nn.Conv2d(encoder_dim, self.proj_dim, kernel_size=1)
        self.decoder_proj = nn.Conv2d(decoder_dim, self.proj_dim, kernel_size=1)
        
        self.norm_encoder = nn.LayerNorm(self.proj_dim)
        self.norm_decoder = nn.LayerNorm(self.proj_dim)
        
        self.cross_attn = nn.MultiheadAttention(self.proj_dim, heads, batch_first=True)
        
        self.output_proj = nn.Conv2d(self.proj_dim, decoder_dim, kernel_size=1)
        
    def forward(self, encoder_features, decoder_features):
        B, C_enc, H_enc, W_enc = encoder_features.shape
        B, C_dec, H_dec, W_dec = decoder_features.shape
        
        if H_enc != H_dec or W_enc != W_dec:
            encoder_features = F.interpolate(encoder_features, size=(H_dec, W_dec), mode='bilinear', align_corners=False)
        
        encoder_proj = self.encoder_proj(encoder_features)  # B, proj_dim, H, W
        decoder_proj = self.decoder_proj(decoder_features)  # B, proj_dim, H, W
        
        encoder_proj = encoder_proj.permute(0, 2, 3, 1)  # B, H, W, proj_dim
        encoder_proj = encoder_proj.reshape(B, H_dec * W_dec, self.proj_dim)
        encoder_proj = self.norm_encoder(encoder_proj)
        
        decoder_proj = decoder_proj.permute(0, 2, 3, 1)  # B, H, W, proj_dim
        decoder_proj = decoder_proj.reshape(B, H_dec * W_dec, self.proj_dim)
        decoder_proj = self.norm_decoder(decoder_proj)
        
        attn_output, _ = self.cross_attn(decoder_proj, encoder_proj, encoder_proj)
        
        attn_output = attn_output.reshape(B, H_dec, W_dec, self.proj_dim)
        attn_output = attn_output.permute(0, 3, 1, 2)  # B, proj_dim, H, W
        
        output = self.output_proj(attn_output)        
        output = output + decoder_features
        
        return output


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class UNet(nn.Module):
    def __init__(self, latent_dim, out_channels, hidden_dim=64, num_heads=8):
        super(UNet, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(latent_dim*2 + 4, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        self.input_proj = nn.Conv2d(latent_dim, hidden_dim, kernel_size=1)
        
        self.enc1_res1 = ResNetBlock(hidden_dim, hidden_dim)
        self.enc1_res2 = ResNetBlock(hidden_dim, hidden_dim)
        self.enc1_maxvit = MaxViTBlock(hidden_dim, heads=num_heads)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2_res1 = ResNetBlock(hidden_dim, hidden_dim*2)
        self.enc2_res2 = ResNetBlock(hidden_dim*2, hidden_dim*2)
        self.enc2_maxvit = MaxViTBlock(hidden_dim*2, heads=num_heads)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3_res1 = ResNetBlock(hidden_dim*2, hidden_dim*4)
        self.enc3_res2 = ResNetBlock(hidden_dim*4, hidden_dim*4)
        self.enc3_maxvit = MaxViTBlock(hidden_dim*4, heads=num_heads)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck_res1 = ResNetBlock(hidden_dim*4, hidden_dim*8)
        self.bottleneck_res2 = ResNetBlock(hidden_dim*8, hidden_dim*8)
        self.bottleneck_maxvit = MaxViTBlock(hidden_dim*8, heads=num_heads)
        
        self.up3 = UpSampleBlock(hidden_dim*8, hidden_dim*4)
        self.cross_attn3 = MaxCrossAttentionBlock(hidden_dim*4, hidden_dim*4, heads=num_heads)
        self.enc3_adj = nn.Conv2d(hidden_dim*4, hidden_dim*4, kernel_size=1)
        self.dec3_res1 = ResNetBlock(hidden_dim*8, hidden_dim*4)  # 8 = 4 (up3) + 4 (enc3)
        self.dec3_res2 = ResNetBlock(hidden_dim*4, hidden_dim*4)
        self.dec3_maxvit = MaxViTBlock(hidden_dim*4, heads=num_heads)
        
        self.up2 = UpSampleBlock(hidden_dim*4, hidden_dim*2)
        self.cross_attn2 = MaxCrossAttentionBlock(hidden_dim*2, hidden_dim*2, heads=num_heads)
        self.enc2_adj = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=1)
        self.dec2_res1 = ResNetBlock(hidden_dim*4, hidden_dim*2)  # 4 = 2 (up2) + 2 (enc2)
        self.dec2_res2 = ResNetBlock(hidden_dim*2, hidden_dim*2)
        self.dec2_maxvit = MaxViTBlock(hidden_dim*2, heads=num_heads)
        
        self.up1 = UpSampleBlock(hidden_dim*2, hidden_dim)
        self.cross_attn1 = MaxCrossAttentionBlock(hidden_dim, hidden_dim, heads=num_heads)
        self.enc1_adj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.dec1_res1 = ResNetBlock(hidden_dim*2, hidden_dim)  # 2 = 1 (up1) + 1 (enc1)
        self.dec1_res2 = ResNetBlock(hidden_dim, hidden_dim)
        self.dec1_maxvit = MaxViTBlock(hidden_dim, heads=num_heads)
        
        self.final_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
    
    def forward(self, x, t, z_0, z_1, m_minus1_to_0, m_0_to_plus1):
        batch_size = x.size(0)
        
        t_emb = self.time_mlp(t)
        t_emb = t_emb.view(batch_size, self.hidden_dim, 1, 1)
        
        z_0 = self._adjust_batch_size(z_0, batch_size)
        z_1 = self._adjust_batch_size(z_1, batch_size)
        m_minus1_to_0 = self._adjust_batch_size(m_minus1_to_0, batch_size)
        m_0_to_plus1 = self._adjust_batch_size(m_0_to_plus1, batch_size)
        
        z_0_resized = F.interpolate(z_0, size=x.shape[2:], mode='bilinear', align_corners=False)
        z_1_resized = F.interpolate(z_1, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        if m_minus1_to_0.dim() == 4 and m_minus1_to_0.shape[1] > 2:
            m_minus1_to_0_resized = m_minus1_to_0[:, :2, :, :]
        else:
            m_minus1_to_0_resized = m_minus1_to_0
            
        if m_0_to_plus1.dim() == 4 and m_0_to_plus1.shape[1] > 2:
            m_0_to_plus1_resized = m_0_to_plus1[:, :2, :, :]
        else:
            m_0_to_plus1_resized = m_0_to_plus1
        
        m_minus1_to_0_resized = F.interpolate(m_minus1_to_0_resized, size=x.shape[2:], mode='bilinear', align_corners=False)
        m_0_to_plus1_resized = F.interpolate(m_0_to_plus1_resized, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        conditions = torch.cat([z_0_resized, z_1_resized, m_minus1_to_0_resized, m_0_to_plus1_resized], dim=1)
        encoded_cond = self.cond_encoder(conditions)
        
        x = self.input_proj(x)
        
        x = x + t_emb + encoded_cond
        
        enc1 = self.enc1_res1(x)
        enc1 = self.enc1_res2(enc1)
        enc1 = self.enc1_maxvit(enc1)
        
        x = self.pool1(enc1)
        
        enc2 = self.enc2_res1(x)
        enc2 = self.enc2_res2(enc2)
        enc2 = self.enc2_maxvit(enc2)
        
        x = self.pool2(enc2)
        
        enc3 = self.enc3_res1(x)
        enc3 = self.enc3_res2(enc3)
        enc3 = self.enc3_maxvit(enc3)
        
        x = self.pool3(enc3)
        
        x = self.bottleneck_res1(x)
        x = self.bottleneck_res2(x)
        x = self.bottleneck_maxvit(x)
        
        x = self.up3(x)
        
        x = self.cross_attn3(enc3, x)
        
        enc3_adjusted = self.enc3_adj(enc3)
        
        if x.shape[2:] != enc3_adjusted.shape[2:]:
            enc3_adjusted = F.interpolate(enc3_adjusted, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, enc3_adjusted], dim=1)
        
        x = self.dec3_res1(x)
        x = self.dec3_res2(x)
        x = self.dec3_maxvit(x)
        
        x = self.up2(x)
        
        x = self.cross_attn2(enc2, x)
        
        enc2_adjusted = self.enc2_adj(enc2)
        
        if x.shape[2:] != enc2_adjusted.shape[2:]:
            enc2_adjusted = F.interpolate(enc2_adjusted, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, enc2_adjusted], dim=1)
        x = self.dec2_res1(x)
        x = self.dec2_res2(x)
        x = self.dec2_maxvit(x)
        
        x = self.up1(x)
        
        x = self.cross_attn1(enc1, x)
        
        enc1_adjusted = self.enc1_adj(enc1)
        
        if x.shape[2:] != enc1_adjusted.shape[2:]:
            enc1_adjusted = F.interpolate(enc1_adjusted, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, enc1_adjusted], dim=1)
        x = self.dec1_res1(x)
        x = self.dec1_res2(x)
        x = self.dec1_maxvit(x)
        
        output = self.final_conv(x)
        
        return output
    
    def _adjust_batch_size(self, tensor, target_batch_size):
        current_batch_size = tensor.size(0)
        
        if current_batch_size == target_batch_size:
            return tensor
            
        if current_batch_size > target_batch_size:
            return tensor[:target_batch_size]
        else:
            repeats_needed = (target_batch_size + current_batch_size - 1) // current_batch_size
            repeated = tensor.repeat(repeats_needed, 1, 1, 1)
            return repeated[:target_batch_size]
    
