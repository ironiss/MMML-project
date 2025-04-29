import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResBlock(nn.Module):
    """Residual Block (ResBlk) as shown in the architecture"""
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(residual)
        out = self.relu(out)
        return out

class ConvBlock(nn.Module):
    """Convolutional Block (Conv) as shown in the architecture"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DownsampleBlock(nn.Module):
    """Downsample Block as shown in the architecture"""
    def __init__(self, scale_factor):
        super(DownsampleBlock, self).__init__()
        self.scale_factor = scale_factor
        
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

class UpsampleBlock(nn.Module):
    """Upsample Block as shown in the architecture"""
    def __init__(self, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.scale_factor = scale_factor
        
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
    
class MAWarpBlock(nn.Module):
    """Motion-Aware Warping Block (MA-Warp) as shown in the architecture"""
    def __init__(self, feature_channels):
        super(MAWarpBlock, self).__init__()
        self.conv1 = nn.Conv2d(feature_channels + 4, feature_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_channels)
        
    def forward(self, features, motion_field):
        x = torch.cat([features, motion_field], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x
    

class VectorQuantizer(nn.Module):
    """Vector Quantizer (VQ-Layer) as shown in the architecture"""
    def __init__(self, num_embeddings=512, embedding_dim=64):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        
        flat_z = z.view(-1, self.embedding_dim)
        
        distances = torch.sum(flat_z**2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight**2, dim=1) - \
                    2 * torch.matmul(flat_z, self.embedding.weight.t())
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        
        quantized_st = z + (quantized - z).detach()
        
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        quantized_st = quantized_st.permute(0, 3, 1, 2).contiguous()
        
        vq_loss = F.mse_loss(quantized.detach(), z.permute(0, 3, 1, 2).contiguous())
        commitment_loss = F.mse_loss(quantized, z.permute(0, 3, 1, 2).detach().contiguous())
        
        return quantized_st, vq_loss, commitment_loss, encoding_indices.view(z.shape[0], z.shape[1], z.shape[2])

    

class Encoder(nn.Module):
    """Encoder architecture as shown in the diagram (top part)"""
    def __init__(self, in_channels=3, hidden_dim=64):
        super(Encoder, self).__init__()
        self.conv_init = ConvBlock(in_channels, hidden_dim)
        
        self.res_blk1 = ResBlock(hidden_dim)
        self.conv1 = ConvBlock(hidden_dim, hidden_dim)
        self.down1 = DownsampleBlock(0.5) 
        
        
        self.res_blk2 = ResBlock(hidden_dim)
        self.conv2 = ConvBlock(hidden_dim, hidden_dim * 2)
        self.down2 = DownsampleBlock(0.5) 
        
        self.res_blk3 = ResBlock(hidden_dim * 2)
        self.conv3 = ConvBlock(hidden_dim * 2, hidden_dim * 4)
        self.down3 = DownsampleBlock(0.5)  
        
        self.res_blk4 = ResBlock(hidden_dim * 4)
        self.conv4 = ConvBlock(hidden_dim * 4, hidden_dim * 8)
        self.down4 = DownsampleBlock(0.5)  
        
        self.res_blk5 = ResBlock(hidden_dim * 8)
        self.conv5 = ConvBlock(hidden_dim * 8, hidden_dim * 8)
        self.down5 = DownsampleBlock(0.5) 
        
        self.res_blk6 = ResBlock(hidden_dim * 8)
        self.conv6 = ConvBlock(hidden_dim * 8, hidden_dim * 8)
        
        self.vq_layer = VectorQuantizer(512, hidden_dim * 8)
        
    def forward(self, x):
        x = self.conv_init(x)
        
        x = self.res_blk1(x)
        x = self.conv1(x)
        x = self.down1(x)
        
        x = self.res_blk2(x)
        x = self.conv2(x)
        x = self.down2(x)
        
        x = self.res_blk3(x)
        x = self.conv3(x)
        x = self.down3(x)
        
        x = self.res_blk4(x)
        x = self.conv4(x)
        x = self.down4(x)
        
        x = self.res_blk5(x)
        x = self.conv5(x)
        x = self.down5(x)
        
        x = self.res_blk6(x)
        enc_features = self.conv6(x)
        
        quantized, vq_loss, commitment_loss, encoding_indices = self.vq_layer(enc_features)
        
        return enc_features, quantized, vq_loss, commitment_loss, encoding_indices
    
class EncoderWithFeatures(nn.Module):
    """Modified Encoder that returns intermediate features"""
    def __init__(self, encoder):
        super(EncoderWithFeatures, self).__init__()
        self.encoder = encoder
        
    def forward(self, x):
        features = {}
        
        x = self.encoder.conv_init(x)
        
        x = self.encoder.res_blk1(x)
        x = self.encoder.conv1(x)
        x = self.encoder.down1(x)
        
        x = self.encoder.res_blk2(x)
        x = self.encoder.conv2(x)
        x = self.encoder.down2(x)
        
        features['block2'] = x.clone()
        
        x = self.encoder.res_blk3(x)
        x = self.encoder.conv3(x)
        x = self.encoder.down3(x)
        
        features['block3'] = x.clone()
        
        x = self.encoder.res_blk4(x)
        x = self.encoder.conv4(x)
        x = self.encoder.down4(x)
        
        features['block4'] = x.clone()
        
        x = self.encoder.res_blk5(x)
        x = self.encoder.conv5(x)
        x = self.encoder.down5(x)
        
        x = self.encoder.res_blk6(x)
        enc_features = self.encoder.conv6(x)
        
        quantized, vq_loss, commitment_loss, encoding_indices = self.encoder.vq_layer(enc_features)
        
        return enc_features, quantized, vq_loss, commitment_loss, encoding_indices, features

class Decoder(nn.Module):
    """Decoder architecture with feature fusion from encoder"""
    def __init__(self, out_channels=3, hidden_dim=64):
        super(Decoder, self).__init__()
        self.initial_conv = ConvBlock(hidden_dim * 8, hidden_dim * 8)
        
        self.fusion_minus1 = nn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=1)
        self.fusion_plus1 = nn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=1)
        
        self.up1 = UpsampleBlock()
        self.ma_warp1 = MAWarpBlock(hidden_dim * 8)
        self.conv1 = ConvBlock(hidden_dim * 8, hidden_dim * 8)
        self.res_blk1 = ResBlock(hidden_dim * 8)
        
        self.up2 = UpsampleBlock()
        self.ma_warp2 = MAWarpBlock(hidden_dim * 8)
        self.conv2 = ConvBlock(hidden_dim * 8, hidden_dim * 4)
        self.res_blk2 = ResBlock(hidden_dim * 4)
        
        self.up3 = UpsampleBlock()
        self.ma_warp3 = MAWarpBlock(hidden_dim * 4)
        
        self.fusion_conv = nn.Conv2d(hidden_dim * 4 + hidden_dim * 4, hidden_dim * 4, kernel_size=3, padding=1)
        self.fusion_bn = nn.BatchNorm2d(hidden_dim * 4)
        self.fusion_relu = nn.ReLU(inplace=True)
        
        self.conv3 = ConvBlock(hidden_dim * 4, hidden_dim * 2)
        self.res_blk3 = ResBlock(hidden_dim * 2)
        
        self.up4 = UpsampleBlock()
        self.ma_warp4 = MAWarpBlock(hidden_dim * 2)
        self.conv4 = ConvBlock(hidden_dim * 2, hidden_dim)
        self.res_blk4 = ResBlock(hidden_dim)
        
        self.up5 = UpsampleBlock()
        self.ma_warp5 = MAWarpBlock(hidden_dim)
        self.conv5 = ConvBlock(hidden_dim, hidden_dim)
        self.res_blk5 = ResBlock(hidden_dim)
        
        self.up6 = UpsampleBlock()
        self.ma_warp6 = MAWarpBlock(hidden_dim)
        self.conv6 = ConvBlock(hidden_dim, hidden_dim)
        self.res_blk6 = ResBlock(hidden_dim)
        
        self.output = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, z, motion_field, phi_minus1=None, phi_plus1=None):
        z = self.initial_conv(z)
        
        m_small = F.interpolate(motion_field, size=z.shape[2:], mode='bilinear', align_corners=False)
        
        x = self.up1(z)
        x = self.ma_warp1(x, F.interpolate(m_small, size=x.shape[2:], mode='bilinear', align_corners=False))
        x = self.conv1(x)
        x = self.res_blk1(x)
        
        x = self.up2(x)
        x = self.ma_warp2(x, F.interpolate(m_small, size=x.shape[2:], mode='bilinear', align_corners=False))
        x = self.conv2(x)
        x = self.res_blk2(x)
        
        x = self.up3(x)
        x = self.ma_warp3(x, F.interpolate(m_small, size=x.shape[2:], mode='bilinear', align_corners=False))
        
        if phi_minus1 is not None and phi_plus1 is not None:
            phi_minus1_processed = self.fusion_minus1(phi_minus1)
            phi_plus1_processed = self.fusion_plus1(phi_plus1)
            
            phi_minus1_resized = F.interpolate(phi_minus1_processed, size=x.shape[2:], mode='bilinear', align_corners=False) 
            phi_plus1_resized = F.interpolate(phi_plus1_processed, size=x.shape[2:], mode='bilinear', align_corners=False)
            
            fused_features = torch.cat([x, phi_minus1_resized, phi_plus1_resized], dim=1)
            
            x = self.fusion_conv(fused_features)
            x = self.fusion_relu(self.fusion_bn(x))
        
        x = self.conv3(x)
        x = self.res_blk3(x)
        
        x = self.up4(x)
        x = self.ma_warp4(x, F.interpolate(m_small, size=x.shape[2:], mode='bilinear', align_corners=False))
        x = self.conv4(x)
        x = self.res_blk4(x)
        
        x = self.up5(x)
        x = self.ma_warp5(x, F.interpolate(m_small, size=x.shape[2:], mode='bilinear', align_corners=False))
        x = self.conv5(x)
        x = self.res_blk5(x)
        
        x = self.up6(x)
        x = self.ma_warp6(x, F.interpolate(m_small, size=x.shape[2:], mode='bilinear', align_corners=False))
        x = self.conv6(x)
        x = self.res_blk6(x)
        
        x = self.tanh(self.output(x))
        
        return x
    
class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for adversarial loss"""
    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super(PatchDiscriminator, self).__init__()
        
        sequence = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                         kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                     kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import lpips

class VQMAGAN(nn.Module):
    """VQ-MAGAN model that combines Encoder, Decoder and EventGAN"""
    def __init__(self, in_channels=3, hidden_dim=64):
        super(VQMAGAN, self).__init__()
        base_encoder = Encoder(in_channels, hidden_dim)
        self.encoder = EncoderWithFeatures(base_encoder)
        self.decoder = Decoder(in_channels, hidden_dim)
        
        self.lpips_loss = lpips.LPIPS(net='alex').to(device)
        
        self.discriminator = PatchDiscriminator(in_channels=3)
        
    def forward(self, I_0, I_minus1, I_plus1, eventgan_model):
        _, z_0, vq_loss_0, commit_loss_0, _, _ = self.encoder(I_0)
        _, z_minus1, vq_loss_minus1, commit_loss_minus1, _, features_minus1 = self.encoder(I_minus1)
        _, z_plus1, vq_loss_plus1, commit_loss_plus1, _, features_plus1 = self.encoder(I_plus1)
        
        with torch.no_grad():  
            I_minus1_gray = 0.299 * I_minus1[:,0,:,:] + 0.587 * I_minus1[:,1,:,:] + 0.114 * I_minus1[:,2,:,:]
            I_0_gray = 0.299 * I_0[:,0,:,:] + 0.587 * I_0[:,1,:,:] + 0.114 * I_0[:,2,:,:]
            I_plus1_gray = 0.299 * I_plus1[:,0,:,:] + 0.587 * I_plus1[:,1,:,:] + 0.114 * I_plus1[:,2,:,:]
            
            I_minus1_gray = I_minus1_gray.unsqueeze(1)
            I_0_gray = I_0_gray.unsqueeze(1)
            I_plus1_gray = I_plus1_gray.unsqueeze(1)
            
            m_minus1_to_0 = eventgan_model(torch.cat([I_minus1_gray, I_0_gray], dim=1))
            m_0_to_plus1 = eventgan_model(torch.cat([I_0_gray, I_plus1_gray], dim=1))
        
        m_minus1_to_0_tensor = m_minus1_to_0[0] if isinstance(m_minus1_to_0, list) else m_minus1_to_0
        m_0_to_plus1_tensor = m_0_to_plus1[0] if isinstance(m_0_to_plus1, list) else m_0_to_plus1
        
        phi_minus1 = features_minus1['block3']  
        phi_plus1 = features_plus1['block3']  
    
        motion_field = torch.cat([
            m_minus1_to_0_tensor[:, :2, :, :], 
            m_0_to_plus1_tensor[:, :2, :, :]   
        ], dim=1)
        
        I_0_pred = self.decoder(z_0, motion_field, phi_minus1, phi_plus1)
        
        vq_loss = vq_loss_0 + vq_loss_minus1 + vq_loss_plus1
        commit_loss = commit_loss_0 + commit_loss_minus1 + commit_loss_plus1
        
        return I_0_pred, vq_loss, commit_loss

from models.eventgan_trainer import EventGANTrainer
from pytorch_utils.base_options import BaseOptions
import torch
import configs

def train_vq_magan():

    
    options = BaseOptions()
    options.parser = configs.get_args(options.parser)
    args = options.parse_args()
    
    eventgan_trainer = EventGANTrainer(options=args, train=False)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        map_location = device
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        map_location = device
        print("Usig Cuda")
    else:
        device = torch.device("cpu")
        map_location = "cpu"
    
    checkpoint_path = "Weights.pt"
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    eventgan_trainer.models_dict["gen"].load_state_dict(checkpoint['gen'])
    eventgan_model = eventgan_trainer.models_dict["gen"].to(device)
    eventgan_model.eval()
    
    vq_magan = VQMAGAN(in_channels=3, hidden_dim=64).to(device)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = FrameTripletDataset(root_dir="sequences_new", transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=7)
    
    optimizer_G = optim.Adam(vq_magan.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(vq_magan.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    
    recon_criterion = nn.L1Loss()
    adversarial_criterion = nn.BCEWithLogitsLoss()
    
    lambda_recon = 10.0   
    lambda_lpips = 1.0 
    lambda_adv = 0.5
    lambda_vq = 1.0
    beta = 0.25
    
    num_epochs = 30
    for epoch in range(num_epochs):
        vq_magan.train()
        running_g_loss = 0.0
        running_d_loss = 0.0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            I_minus1 = batch["I_minus1"].to(device)
            I_0 = batch["I_0"].to(device)
            I_plus1 = batch["I_plus1"].to(device)
            
            optimizer_D.zero_grad()
            
            I_0_pred, vq_loss, commit_loss = vq_magan(I_0, I_minus1, I_plus1, eventgan_model)
            I_0_pred_resized = F.interpolate(I_0_pred, size=I_0.shape[2:], mode='bilinear', align_corners=False)
            
            real_pred = vq_magan.discriminator(I_0)
            real_target = torch.ones_like(real_pred).to(device)
            d_real_loss = adversarial_criterion(real_pred, real_target)
            
            fake_pred = vq_magan.discriminator(I_0_pred_resized.detach())
            fake_target = torch.zeros_like(fake_pred).to(device)
            d_fake_loss = adversarial_criterion(fake_pred, fake_target)
            
            d_loss = (d_real_loss + d_fake_loss) * 0.5 
            d_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(vq_magan.discriminator.parameters(), max_norm=1.0)
            
            optimizer_D.step()
            running_d_loss += d_loss.item()
            
            optimizer_G.zero_grad()
            
            I_0_pred, vq_loss, commit_loss = vq_magan(I_0, I_minus1, I_plus1, eventgan_model)
            I_0_pred_resized = F.interpolate(I_0_pred, size=I_0.shape[2:], mode='bilinear', align_corners=False)
            
            recon_loss = recon_criterion(I_0_pred_resized, I_0)
            I_0_pred_resized_normalized = I_0_pred_resized / 255.0 * 2 - 1
            I_0_normalized = I_0 / 255.0 * 2 - 1
            lpips_loss = vq_magan.lpips_loss(I_0_pred_resized_normalized, I_0_normalized).mean()
            
            gen_pred = vq_magan.discriminator(I_0_pred_resized)
            gen_target = torch.ones_like(gen_pred).to(device)
            g_adv_loss = adversarial_criterion(gen_pred, gen_target)
            
            g_loss = (lambda_recon * recon_loss + 
                     lambda_lpips * lpips_loss + 
                     lambda_adv * g_adv_loss + 
                     lambda_vq * vq_loss + 
                     beta * commit_loss)
            
            g_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(vq_magan.parameters(), max_norm=1.0)
            
            optimizer_G.step()
            
            running_g_loss += g_loss.item()
        
        epoch_g_loss = running_g_loss / len(dataloader)
        epoch_d_loss = running_d_loss / len(dataloader)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  G Loss: {epoch_g_loss:.4f}")
        print(f"  D Loss: {epoch_d_loss:.4f}")
        
        if epoch % 5 == 0:
            with torch.no_grad():
                sample_batch = next(iter(dataloader))
                I_minus1 = sample_batch["I_minus1"].to(device)[:1]
                I_0 = sample_batch["I_0"].to(device)[:1]
                I_plus1 = sample_batch["I_plus1"].to(device)[:1]
                
                I_0_pred, vq_loss, commit_loss = vq_magan(I_0, I_minus1, I_plus1, eventgan_model)
                I_0_pred_resized = F.interpolate(I_0_pred, size=I_0.shape[2:], mode='bilinear', align_corners=False)
                
                recon_loss = recon_criterion(I_0_pred_resized, I_0).item()
                lpips_loss = vq_magan.lpips_loss(I_0_pred_resized, I_0).mean().item()
                
                print(f"  Detailed metrics:")
                print(f"    Reconstruction loss: {recon_loss:.4f}")
                print(f"    LPIPS loss: {lpips_loss:.4f}")
                print(f"    VQ loss: {vq_loss.item():.4f}")
                print(f"    Commitment loss: {commit_loss.item():.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': vq_magan.encoder.encoder.state_dict(),
                'decoder_state_dict': vq_magan.decoder.state_dict(),
                'discriminator_state_dict': vq_magan.discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'g_loss': epoch_g_loss,
                'd_loss': epoch_d_loss
            }, f"vq_magan_checkpoint_epoch{epoch+1}.pt")

def calculate_perceptual_loss(pred, target):
    return F.mse_loss(pred, target)


import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class FrameTripletDataset(Dataset):
    """Dataset class for loading frame triplets from nested folder structure"""
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Path to the sequences folder (contains 00001, 00002, etc.)
            transform: Optional transform to be applied to frames
        """
        self.root_dir = root_dir
        self.transform = transform
        self.triplets = []
        
        for seq_folder in sorted(os.listdir(root_dir)):
            seq_path = os.path.join(root_dir, seq_folder)
            if os.path.isdir(seq_path):
                
                for sub_folder in sorted(os.listdir(seq_path)):
                    sub_path = os.path.join(seq_path, sub_folder)
                    if os.path.isdir(sub_path):
                        
                        frames = sorted([f for f in os.listdir(sub_path) 
                                       if f.startswith('im') and (f.endswith('.png') or f.endswith('.jpg'))])
                        
                        if len(frames) >= 3:
                            self.triplets.append([
                                os.path.join(sub_path, 'im1.png'),  
                                os.path.join(sub_path, 'im2.png'),  
                                os.path.join(sub_path, 'im3.png') 
                            ])
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        prev_path, target_path, next_path = self.triplets[idx]
        
        I_prev = Image.open(prev_path).convert('RGB')
        I_target = Image.open(target_path).convert('RGB')
        I_next = Image.open(next_path).convert('RGB')
        
        if self.transform:
            I_prev = self.transform(I_prev)
            I_target = self.transform(I_target)
            I_next = self.transform(I_next)
        
        return {
            'I_minus1': I_prev,    
            'I_0': I_target, 
            'I_plus1': I_next      
        }
    
from torchvision import transforms
import multiprocessing

def main():
    train_vq_magan()

if __name__ == '__main__':
    multiprocessing.freeze_support() 
    main()