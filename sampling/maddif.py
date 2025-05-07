
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import multiprocessing
from torchvision.utils import save_image
import math
from .unet import UNet, device

import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
import configs

from training_encod_decod.VQMAGAN import VQMAGAN
from EVENTGAN.models.eventgan_trainer import EventGANTrainer
from EVENTGAN.pytorch_utils.base_options import BaseOptions
from training_encod_decod.dataset_fix import FrameTripletDataset



class MADiff(nn.Module):
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
        batch_size = I_minus1.shape[0]
        device = I_minus1.device
        
        z_minus1, indices_minus1, emb_minus1, z_q_minus1, phi_minus1, features_minus1 = self.vq_magan.encoder(I_minus1)
        z_plus1, indices_plus1, emb_plus1, z_q_plus1, phi_plus1, features_plus1 = self.vq_magan.encoder(I_plus1)
        
        z_T = torch.randn(batch_size, self.latent_dim,
                        z_minus1.shape[2], z_minus1.shape[3],
                        device=device)
        
        z_t = z_T
        
        expected_channels = None
        for name, module in self.vq_magan.decoder.named_modules():
            if isinstance(module, nn.Conv2d) and name == 'initial_conv.conv': 
                expected_channels = module.in_channels
                print(f"Decoder's initial conv expects {expected_channels} input channels")
                break
        
        if expected_channels is None:
            expected_channels = 516  
            print(f"Using fallback channel count: {expected_channels}")
        
        I_0_t = (I_minus1 + I_plus1) / 2.0 
        
        with torch.no_grad():
            I_minus1_gray = 0.299 * I_minus1[:,0,:,:] + 0.587 * I_minus1[:,1,:,:] + 0.114 * I_minus1[:,2,:,:]
            I_0_t_gray = 0.299 * I_0_t[:,0,:,:] + 0.587 * I_0_t[:,1,:,:] + 0.114 * I_0_t[:,2,:,:]
            I_plus1_gray = 0.299 * I_plus1[:,0,:,:] + 0.587 * I_plus1[:,1,:,:] + 0.114 * I_plus1[:,2,:,:]
            
            I_minus1_gray = I_minus1_gray.unsqueeze(1)
            I_0_t_gray = I_0_t_gray.unsqueeze(1)
            I_plus1_gray = I_plus1_gray.unsqueeze(1)
            
            target_size = 256
            I_minus1_gray_resized = F.interpolate(I_minus1_gray, size=(target_size, target_size), mode='bilinear', align_corners=False)
            I_0_t_gray_resized = F.interpolate(I_0_t_gray, size=(target_size, target_size), mode='bilinear', align_corners=False)
            I_plus1_gray_resized = F.interpolate(I_plus1_gray, size=(target_size, target_size), mode='bilinear', align_corners=False)
            
            m_output_minus1_to_0 = eventgan_model(torch.cat([I_minus1_gray_resized, I_0_t_gray_resized], dim=1))
            m_output_0_to_plus1 = eventgan_model(torch.cat([I_0_t_gray_resized, I_plus1_gray_resized], dim=1))
            
            m_minus1_to_0 = m_output_minus1_to_0[0] if isinstance(m_output_minus1_to_0, list) else m_output_minus1_to_0
            m_0_to_plus1 = m_output_0_to_plus1[0] if isinstance(m_output_0_to_plus1, list) else m_output_0_to_plus1
            
            if m_minus1_to_0.shape[2:] != z_t.shape[2:]:
                m_minus1_to_0 = F.interpolate(m_minus1_to_0, size=z_t.shape[2:], mode='bilinear', align_corners=False)
            if m_0_to_plus1.shape[2:] != z_t.shape[2:]:
                m_0_to_plus1 = F.interpolate(m_0_to_plus1, size=z_t.shape[2:], mode='bilinear', align_corners=False)
        
        for t in range(steps-1, -1, -1):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            predicted_noise = self.denoising_unet(z_t, t_tensor, z_minus1, z_plus1, m_minus1_to_0, m_0_to_plus1)
            
            if predicted_noise.shape != z_t.shape:
                predicted_noise = F.interpolate(predicted_noise, size=z_t.shape[2:], mode='bilinear', align_corners=False)
            
            alpha_t = self.alphas_cumprod[t]
            alpha_t = alpha_t.view(-1, 1, 1, 1)
            noise_coefficient = (1 - alpha_t) / torch.sqrt(1 - alpha_t)
            z_0_t = (z_t - noise_coefficient * predicted_noise) / torch.sqrt(alpha_t)
            
            if z_0_t.shape[1] != expected_channels:
                adjusted_z_0_t = torch.zeros(z_0_t.shape[0], expected_channels, 
                                        z_0_t.shape[2], z_0_t.shape[3],
                                        device=z_0_t.device)
                min_channels = min(z_0_t.shape[1], expected_channels)
                adjusted_z_0_t[:, :min_channels, :, :] = z_0_t[:, :min_channels, :, :]
                z_0_t = adjusted_z_0_t
            
            try:
                motion_field = torch.cat([
                    m_minus1_to_0[:, :2, :, :],
                    m_0_to_plus1[:, :2, :, :]
                ], dim=1)
                
                phi_minus1_block3 = features_minus1['block3'] if 'block3' in features_minus1 else phi_minus1
                phi_plus1_block3 = features_plus1['block3'] if 'block3' in features_plus1 else phi_plus1
                
                I_0_t = self.vq_magan.decoder(z_0_t, motion_field, phi_minus1_block3, phi_plus1_block3)
                
            except Exception as e:
                print(f"Error in frame reconstruction: {e}")
                try:
                    default_motion_field = torch.zeros(
                        batch_size, 4, z_t.shape[2], z_t.shape[3], device=device
                    )
                    I_0_t = self.vq_magan.decoder(z_0_t, default_motion_field)
                except Exception as e2:
                    print(f"Fallback decoder call failed: {e2}")
                    I_0_t = (I_minus1 + I_plus1) / 2.0
            
            if t > 0:
                with torch.no_grad():
                    I_0_t_gray = 0.299 * I_0_t[:,0,:,:] + 0.587 * I_0_t[:,1,:,:] + 0.114 * I_0_t[:,2,:,:]
                    I_0_t_gray = I_0_t_gray.unsqueeze(1)
                    
                    target_size = 256 
                    I_0_t_gray_resized = F.interpolate(I_0_t_gray, size=(target_size, target_size), mode='bilinear', align_corners=False)
                    
                    m_output_minus1_to_0 = eventgan_model(torch.cat([I_minus1_gray_resized, I_0_t_gray_resized], dim=1))
                    m_output_0_to_plus1 = eventgan_model(torch.cat([I_0_t_gray_resized, I_plus1_gray_resized], dim=1))
                    
                    m_minus1_to_0 = m_output_minus1_to_0[0] if isinstance(m_output_minus1_to_0, list) else m_output_minus1_to_0
                    m_0_to_plus1 = m_output_0_to_plus1[0] if isinstance(m_output_0_to_plus1, list) else m_output_0_to_plus1
                    
                    if m_minus1_to_0.shape[2:] != z_t.shape[2:]:
                        m_minus1_to_0 = F.interpolate(m_minus1_to_0, size=z_t.shape[2:], mode='bilinear', align_corners=False)
                    if m_0_to_plus1.shape[2:] != z_t.shape[2:]:
                        m_0_to_plus1 = F.interpolate(m_0_to_plus1, size=z_t.shape[2:], mode='bilinear', align_corners=False)
                
                sigma_t = torch.sqrt(self.betas[t])
                sigma_t = sigma_t.view(-1, 1, 1, 1)
                
                noise = torch.randn_like(z_t)
                
                if t > 1:
                    alpha_t_minus_1 = self.alphas_cumprod[t-1]
                    alpha_t_minus_1 = alpha_t_minus_1.view(-1, 1, 1, 1)
                    
                    posterior_mean = (
                        torch.sqrt(alpha_t_minus_1) * 
                        (z_t - (1 - alpha_t)/torch.sqrt(1 - alpha_t) * predicted_noise) +
                        torch.sqrt(1 - alpha_t_minus_1) * predicted_noise
                    )
                    
                    z_t = posterior_mean + sigma_t * noise
                else:
                    z_t = z_0_t + sigma_t * noise
        
        if z_0_t.shape[1] != expected_channels:
            adjusted_z_0_t = torch.zeros(z_0_t.shape[0], expected_channels, 
                                    z_0_t.shape[2], z_0_t.shape[3],
                                    device=z_0_t.device)
            min_channels = min(z_0_t.shape[1], expected_channels)
            adjusted_z_0_t[:, :min_channels, :, :] = z_0_t[:, :min_channels, :, :]
            z_0_t = adjusted_z_0_t
            print(f"Adjusted final z_0_t to exactly {expected_channels} channels")
        
        motion_field = torch.cat([
            m_minus1_to_0[:, :2, :, :],
            m_0_to_plus1[:, :2, :, :]
        ], dim=1)
        
        phi_minus1_block3 = features_minus1['block3'] if 'block3' in features_minus1 else phi_minus1
        phi_plus1_block3 = features_plus1['block3'] if 'block3' in features_plus1 else phi_plus1
        
        try:
            final_frame = self.vq_magan.decoder(z_0_t, motion_field, phi_minus1_block3, phi_plus1_block3)
        except Exception as e:
            print(f"Error in final frame reconstruction: {e}")
            try:
                default_motion_field = torch.zeros(
                    batch_size, 4, z_t.shape[2], z_t.shape[3], device=device
                )
                final_frame = self.vq_magan.decoder(z_0_t, default_motion_field)
            except Exception as e2:
                print(f"All reconstruction approaches failed: {e2}")
                final_frame = (I_minus1 + I_plus1) / 2.0
        
        return final_frame

def train_madiff(vq_magan_checkpoint_path, eventgan_checkpoint_path):
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
    
    dataset = FrameTripletDataset(root_dir="sampling/sequences", transform=transform)
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
                
                save_image(I_0_pred[0], f"madiff_sample_epoch{epoch+1}.png")
                save_image(I_0_val[0], f"madiff_sample_val_epoch{epoch+1}.png")

def main():
    vq_magan_checkpoint_path = "sampling/vq_magan_checkpoint_epoch20.pt"
    eventgan_checkpoint_path = "sampling/Weights.pt"
    
    train_madiff(vq_magan_checkpoint_path, eventgan_checkpoint_path)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()