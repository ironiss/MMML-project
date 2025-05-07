import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lpips
import configs
import os
import multiprocessing


from torchvision import transforms
from tqdm import tqdm


from EVENTGAN.models.eventgan_trainer import EventGANTrainer
from EVENTGAN.pytorch_utils.base_options import BaseOptions
from dataset_fix import FrameTripletDataset

from torch.utils.data import DataLoader
from torchvision import transforms
from VQMAGAN import VQMAGAN


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
    
    num_epochs = 1
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


def main():
    train_vq_magan()

if __name__ == '__main__':
    multiprocessing.freeze_support() 
    main()