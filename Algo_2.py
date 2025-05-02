
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import lpips
from Algo_1 import VQMAGAN
from models.eventgan_trainer import EventGANTrainer
from pytorch_utils.base_options import BaseOptions
import torch
import configs
from Algo_1 import FrameTripletDataset
import torchvision.transforms as transforms
import multiprocessing
from torchvision.utils import save_image
import inspect
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNetBlock(nn.Module):
    """ResNet block with residual connection"""
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
    """MaxViT block combining MBConv with spatial and channel attention"""
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
        x_spatial = x.permute(0, 2, 3, 1)  
        x_spatial = x_spatial.reshape(B, H * W, C)
        x_spatial = self.spatial_norm(x_spatial)
        x_spatial, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        x_spatial = x_spatial.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + x_spatial
        
        residual = x
        x_channel = x.reshape(B, C, H * W).permute(0, 2, 1) 
        x_channel = self.channel_norm(x_channel)
        x_channel, _ = self.channel_attn(x_channel, x_channel, x_channel)
        x_channel = x_channel.permute(0, 2, 1).reshape(B, C, H, W)
        x = x + x_channel
        
        residual = x
        x_mlp = x.permute(0, 2, 3, 1)  
        x_mlp = x_mlp.reshape(B, H * W, C)
        x_mlp = self.mlp_norm(x_mlp)
        x_mlp = self.mlp(x_mlp)
        x_mlp = x_mlp.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + x_mlp
        
        return x

class MaxCrossAttentionBlock(nn.Module):
    """MaxViT Cross-Attention Block for U-Net skip connections"""
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
        
        encoder_proj = self.encoder_proj(encoder_features)  
        decoder_proj = self.decoder_proj(decoder_features) 


        encoder_proj = encoder_proj.permute(0, 2, 3, 1) 
        encoder_proj = encoder_proj.reshape(B, H_dec * W_dec, self.proj_dim)
        encoder_proj = self.norm_encoder(encoder_proj)
        
        decoder_proj = decoder_proj.permute(0, 2, 3, 1)  
        decoder_proj = decoder_proj.reshape(B, H_dec * W_dec, self.proj_dim)
        decoder_proj = self.norm_decoder(decoder_proj)
        
        attn_output, _ = self.cross_attn(decoder_proj, encoder_proj, encoder_proj)
        
        attn_output = attn_output.reshape(B, H_dec, W_dec, self.proj_dim)
        attn_output = attn_output.permute(0, 3, 1, 2)  
        
        output = self.output_proj(attn_output)
        
        output = output + decoder_features
        
        return output

class UpSampleBlock(nn.Module):
    """Up-sampling block for the decoder path"""
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings similar to those used in the transformer model"""
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
    """Advanced U-Net with time-conditioning and MaxViT blocks - fixed for tensor shape compatibility"""
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
        self.dec3_res1 = ResNetBlock(hidden_dim*8, hidden_dim*4)  
        self.dec3_res2 = ResNetBlock(hidden_dim*4, hidden_dim*4)
        self.dec3_maxvit = MaxViTBlock(hidden_dim*4, heads=num_heads)
        
        self.up2 = UpSampleBlock(hidden_dim*4, hidden_dim*2)
        self.cross_attn2 = MaxCrossAttentionBlock(hidden_dim*2, hidden_dim*2, heads=num_heads)

        self.enc2_adj = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=1)
        self.dec2_res1 = ResNetBlock(hidden_dim*4, hidden_dim*2) 
        self.dec2_res2 = ResNetBlock(hidden_dim*2, hidden_dim*2)
        self.dec2_maxvit = MaxViTBlock(hidden_dim*2, heads=num_heads)
        
        self.up1 = UpSampleBlock(hidden_dim*2, hidden_dim)
        self.cross_attn1 = MaxCrossAttentionBlock(hidden_dim, hidden_dim, heads=num_heads)

        self.enc1_adj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.dec1_res1 = ResNetBlock(hidden_dim*2, hidden_dim) 
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
        """Adjust the batch size of a tensor to match the target batch size"""
        current_batch_size = tensor.size(0)
        
        if current_batch_size == target_batch_size:
            return tensor
            
        if current_batch_size > target_batch_size:
            return tensor[:target_batch_size]
        else:
            repeats_needed = (target_batch_size + current_batch_size - 1) // current_batch_size
            repeated = tensor.repeat(repeats_needed, 1, 1, 1)
            return repeated[:target_batch_size]
    

class MADiff(nn.Module):
    """MADiff model that implements Algorithm 2"""
    def __init__(self, vq_magan, max_diffusion_steps=1000):
        super(MADiff, self).__init__()
        self.vq_magan = vq_magan
        self.encoder = vq_magan.encoder.encoder
        
        self.latent_dim = self.encoder.vq_layer.embedding_dim
        
        self.denoising_unet = UNet(
            latent_dim=self.latent_dim,
            out_channels=self.latent_dim,
            hidden_dim=64
        )
        self.max_diffusion_steps = max_diffusion_steps
        
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.betas = torch.linspace(self.beta_start, self.beta_end, max_diffusion_steps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
    def forward(self, I_minus1, I_0, I_plus1, eventgan_model, t, epsilon=None):
        """Forward pass for training"""
        z_minus1, _, _, _, _, _ = self.vq_magan.encoder(I_minus1)
        z_0, _, _, _, _, _ = self.vq_magan.encoder(I_0)
        z_plus1, _, _, _, _, _ = self.vq_magan.encoder(I_plus1)
        
        batch_size = z_0.shape[0]
        
        with torch.no_grad():
            I_minus1_gray = 0.299 * I_minus1[:,0,:,:] + 0.587 * I_minus1[:,1,:,:] + 0.114 * I_minus1[:,2,:,:]
            I_0_gray = 0.299 * I_0[:,0,:,:] + 0.587 * I_0[:,1,:,:] + 0.114 * I_0[:,2,:,:]
            I_plus1_gray = 0.299 * I_plus1[:,0,:,:] + 0.587 * I_plus1[:,1,:,:] + 0.114 * I_plus1[:,2,:,:]
            
            I_minus1_gray = I_minus1_gray.unsqueeze(1)
            I_0_gray = I_0_gray.unsqueeze(1)
            I_plus1_gray = I_plus1_gray.unsqueeze(1)
            
            m_output_minus1_to_0 = eventgan_model(torch.cat([I_minus1_gray, I_0_gray], dim=1))
            m_output_0_to_plus1 = eventgan_model(torch.cat([I_0_gray, I_plus1_gray], dim=1))
            
            if isinstance(m_output_minus1_to_0, list):
                m_minus1_to_0 = m_output_minus1_to_0[0]
            else:
                m_minus1_to_0 = m_output_minus1_to_0
                
            if isinstance(m_output_0_to_plus1, list):
                m_0_to_plus1 = m_output_0_to_plus1[0]
            else:
                m_0_to_plus1 = m_output_0_to_plus1
            
        if epsilon is None:
            epsilon = torch.randn_like(z_0)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        z_t = sqrt_alphas_cumprod_t * z_0 + sqrt_one_minus_alphas_cumprod_t * epsilon
        
        if z_0.size(0) != batch_size:
            z_0 = self._adjust_batch_size(z_0, batch_size)
        if z_plus1.size(0) != batch_size:
            z_plus1 = self._adjust_batch_size(z_plus1, batch_size)
        if m_minus1_to_0.size(0) != batch_size:
            m_minus1_to_0 = self._adjust_batch_size(m_minus1_to_0, batch_size)
        if m_0_to_plus1.size(0) != batch_size:
            m_0_to_plus1 = self._adjust_batch_size(m_0_to_plus1, batch_size)
        if z_t.size(0) != batch_size:
            z_t = self._adjust_batch_size(z_t, batch_size)
        
        predicted_noise = self.denoising_unet(z_t, t, z_0, z_plus1, m_minus1_to_0, m_0_to_plus1)
        
        if predicted_noise.shape != epsilon.shape:
            epsilon = F.interpolate(epsilon, size=predicted_noise.shape[2:], mode='bilinear', align_corners=False)
        
        return predicted_noise, epsilon

    def _adjust_batch_size(self, tensor, target_batch_size):
        """Adjust the batch size of a tensor to match the target batch size"""
        current_batch_size = tensor.size(0)
        
        if current_batch_size == target_batch_size:
            return tensor
            
        if current_batch_size > target_batch_size:
            return tensor[:target_batch_size]
        else:
            repeats_needed = (target_batch_size + current_batch_size - 1) // current_batch_size
            repeated = tensor.repeat(repeats_needed, 1, 1, 1)
            return repeated[:target_batch_size]
    
    @torch.no_grad()
    def sample(self, I_minus1, I_plus1, eventgan_model, steps=100):
        """Sample a new I_0 frame given I_minus1 and I_plus1"""
        batch_size = I_minus1.shape[0]
        
        z_minus1, _, _, _, _, _ = self.vq_magan.encoder(I_minus1)
        z_plus1, _, _, _, _, _ = self.vq_magan.encoder(I_plus1)
        
        I_minus1_gray = 0.299 * I_minus1[:,0,:,:] + 0.587 * I_minus1[:,1,:,:] + 0.114 * I_minus1[:,2,:,:]
        I_plus1_gray = 0.299 * I_plus1[:,0,:,:] + 0.587 * I_plus1[:,1,:,:] + 0.114 * I_plus1[:,2,:,:]
        
        I_minus1_gray = I_minus1_gray.unsqueeze(1)
        I_plus1_gray = I_plus1_gray.unsqueeze(1)
        
        I_0_fake = (I_minus1 + I_plus1) / 2.0
        I_0_gray = 0.299 * I_0_fake[:,0,:,:] + 0.587 * I_0_fake[:,1,:,:] + 0.114 * I_0_fake[:,2,:,:]
        I_0_gray = I_0_gray.unsqueeze(1)
        
        I_0_gray_updated = I_0_gray.clone()
        
        m_output_minus1_to_0 = eventgan_model(torch.cat([I_minus1_gray, I_0_gray], dim=1))
        m_output_0_to_plus1 = eventgan_model(torch.cat([I_0_gray, I_plus1_gray], dim=1))
        
        if isinstance(m_output_minus1_to_0, list):
            m_minus1_to_0 = m_output_minus1_to_0[0]
        else:
            m_minus1_to_0 = m_output_minus1_to_0
            
        if isinstance(m_output_0_to_plus1, list):
            m_0_to_plus1 = m_output_0_to_plus1[0]
        else:
            m_0_to_plus1 = m_output_0_to_plus1
        
        motion_field = torch.cat([
            m_minus1_to_0[:, :2, :, :],
            m_0_to_plus1[:, :2, :, :]
        ], dim=1)
        
        z_t = torch.randn(batch_size, self.latent_dim, 
                        z_minus1.shape[2], z_minus1.shape[3], 
                        device=I_minus1.device) 
        
        z_0 = torch.zeros_like(z_t)
        
        for time_step in range(steps-1, -1, -1):
            t = torch.full((batch_size,), time_step, device=I_minus1.device, dtype=torch.long)
            
            predicted_noise = self.denoising_unet(z_t, t, z_0, z_plus1, m_minus1_to_0, m_0_to_plus1)
            
            alpha = self.alphas[time_step]
            alpha_cumprod = self.alphas_cumprod[time_step]
            beta = self.betas[time_step]
            
            if time_step > 0:
                noise = torch.randn_like(z_t)
                alpha_cumprod_prev = self.alphas_cumprod[time_step-1]
            else:
                noise = torch.zeros_like(z_t)
                alpha_cumprod_prev = torch.tensor(1.0, device=I_minus1.device)
            
            z_0 = (z_t - self.sqrt_one_minus_alphas_cumprod[time_step].to(I_minus1.device).view(-1, 1, 1, 1) * predicted_noise) / self.sqrt_alphas_cumprod[time_step].to(I_minus1.device).view(-1, 1, 1, 1)
            
            posterior_mean_coef1 = torch.sqrt(alpha_cumprod_prev) * beta / (1 - alpha_cumprod)
            posterior_mean_coef2 = torch.sqrt(alpha) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
            
            posterior_mean = posterior_mean_coef1 * z_0 + posterior_mean_coef2 * z_t
            posterior_var = beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
            z_t = posterior_mean + torch.sqrt(posterior_var) * noise
            
            if time_step % 20 == 0 and time_step > 0:
                with torch.no_grad():
                    try:
                        I_0_updated = self.vq_magan.decoder(z_0, motion_field)
                    except TypeError:
                        try:
                            I_0_updated = self.vq_magan.decoder(z_0)
                        except Exception as e:
                            print(f"Error decoding z_0: {e}")
                            continue
                    
                    I_0_gray_updated = 0.299 * I_0_updated[:,0,:,:] + 0.587 * I_0_updated[:,1,:,:] + 0.114 * I_0_updated[:,2,:,:]
                    I_0_gray_updated = I_0_gray_updated.unsqueeze(1)
                    
                    if I_minus1_gray.shape != I_0_gray_updated.shape:
                        I_0_gray_updated = F.interpolate(
                            I_0_gray_updated,
                            size=(I_minus1_gray.shape[2], I_minus1_gray.shape[3]),
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    m_output_minus1_to_0 = eventgan_model(torch.cat([I_minus1_gray, I_0_gray_updated], dim=1))
                    m_output_0_to_plus1 = eventgan_model(torch.cat([I_0_gray_updated, I_plus1_gray], dim=1))
                
                    if isinstance(m_output_minus1_to_0, list):
                        m_minus1_to_0 = m_output_minus1_to_0[0]
                    else:
                        m_minus1_to_0 = m_output_minus1_to_0
                        
                    if isinstance(m_output_0_to_plus1, list):
                        m_0_to_plus1 = m_output_0_to_plus1[0]
                    else:
                        m_0_to_plus1 = m_output_0_to_plus1
                    
                    motion_field = torch.cat([
                        m_minus1_to_0[:, :2, :, :],
                        m_0_to_plus1[:, :2, :, :]
                    ], dim=1)
        
        try:
            I_0_pred = self.vq_magan.decoder(z_t, motion_field)
        except TypeError:
            try:
                I_0_pred = self.vq_magan.decoder(z_t)
            except Exception as e:
                print(f"Final decoding failed: {e}")
                I_0_pred = (I_minus1 + I_plus1) / 2.0
        
        return I_0_pred

def train_madiff(vq_magan_checkpoint_path, eventgan_checkpoint_path):
    """Training procedure for MADiff (Algorithm 2)"""
    vq_magan = VQMAGAN(in_channels=3, hidden_dim=64).to(device)
    checkpoint = torch.load(vq_magan_checkpoint_path, map_location=device)
    vq_magan.encoder.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    vq_magan.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    vq_magan.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    options = BaseOptions()
    options.parser = configs.get_args(options.parser)
    args = options.parse_args()
    eventgan_trainer = EventGANTrainer(options=args, train=False)
    eventgan_checkpoint = torch.load(eventgan_checkpoint_path, map_location=device)
    eventgan_trainer.models_dict["gen"].load_state_dict(eventgan_checkpoint['gen'])
    eventgan_model = eventgan_trainer.models_dict["gen"].to(device)
    eventgan_model.eval()
    
    madiff = MADiff(vq_magan, max_diffusion_steps=1000).to(device)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = FrameTripletDataset(root_dir="sequences", transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    optimizer = optim.Adam(madiff.denoising_unet.parameters(), lr=1e-4)
    
    max_steps = 1000
    
    num_epochs = 100
    for epoch in range(num_epochs):
        madiff.train()
        running_loss = 0.0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            I_minus1 = batch["I_minus1"].to(device)
            I_0 = batch["I_0"].to(device)
            I_plus1 = batch["I_plus1"].to(device)
            
            batch_size = I_0.shape[0]
            t = torch.randint(0, max_steps, (batch_size,), device=device)
            
            with torch.no_grad():
                z_0, _, _, _, _, _ = madiff.vq_magan.encoder(I_0)
                epsilon = torch.randn_like(z_0).to(device)
            
            optimizer.zero_grad()
            

            predicted_noise, target_noise = madiff(I_minus1, I_0, I_plus1, eventgan_model, t, epsilon)
        
            loss = F.mse_loss(predicted_noise, target_noise)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            torch.save({
                'epoch': epoch,
                'denoising_unet_state_dict': madiff.denoising_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
            }, f"madiff_checkpoint_epoch{epoch+1}.pt")
            
            madiff.eval()
            with torch.no_grad():
                val_batch = next(iter(dataloader))
                I_minus1_val = val_batch["I_minus1"].to(device)
                I_0_val = val_batch["I_0"].to(device)
                I_plus1_val = val_batch["I_plus1"].to(device)
                
                I_0_pred = madiff.sample(I_minus1_val, I_plus1_val, eventgan_model, steps=100)
                
                save_image(I_0_pred, f"madiff_sample_epoch{epoch+1}.png")
                save_image(I_0_val, f"madiff_sample_val_epoch{epoch+1}.png")

def main():
    vq_magan_checkpoint_path = "vq_magan_checkpoint_epoch20.pt"
    eventgan_checkpoint_path = "Weights.pt"
    
    train_madiff(vq_magan_checkpoint_path, eventgan_checkpoint_path)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()