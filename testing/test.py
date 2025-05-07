import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import sys

import os.path as osp
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))

from tqdm import tqdm
from training_encod_decod.VQMAGAN import VQMAGAN
from EVENTGAN.models.eventgan_trainer import EventGANTrainer
from EVENTGAN.pytorch_utils.base_options import BaseOptions
import configs
from training_encod_decod.dataset_fix import FrameTripletDataset
from sampling.maddif import MADiff

def test_madiff():
    madiff_weights_path = "madiff_checkpoint_epoch70.pt"  
    vqmagan_weights_path = "vq_magan_checkpoint_epoch20.pt"  
    eventgan_weights_path = "Weights.pt" 
    test_data_dir = "sequences"  
    output_dir = "generated"
    
    batch_size = 1  
    diffusion_steps = 100  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    os.makedirs(output_dir, exist_ok=True)
    
    vq_magan = VQMAGAN(in_channels=3, hidden_dim=64).to(device)
    checkpoint = torch.load(vqmagan_weights_path, map_location=device)
    vq_magan.encoder.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    vq_magan.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    vq_magan.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    vq_magan.eval()
    
    options = BaseOptions()
    options.parser = configs.get_args(options.parser)
    config_args = options.parse_args()
    eventgan_trainer = EventGANTrainer(options=config_args, train=False)
    eventgan_checkpoint = torch.load(eventgan_weights_path, map_location=device)
    eventgan_trainer.models_dict["gen"].load_state_dict(eventgan_checkpoint['gen'])
    eventgan_model = eventgan_trainer.models_dict["gen"].to(device)
    eventgan_model.eval()
    
    madiff = MADiff(vq_magan, max_diffusion_steps=1000).to(device)
    madiff_checkpoint = torch.load(madiff_weights_path, map_location=device)
    madiff.denoising_unet.load_state_dict(madiff_checkpoint['denoising_unet_state_dict'])
    madiff.eval()
    
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    test_dataset = FrameTripletDataset(root_dir=test_data_dir, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Testing MADiff")):
            I_minus1 = batch["I_minus1"].to(device)
            I_0 = batch["I_0"].to(device) 
            I_plus1 = batch["I_plus1"].to(device)
            
            I_0_pred = madiff.sample(I_minus1, I_plus1, eventgan_model, steps=diffusion_steps)
            
            for i in range(I_0_pred.shape[0]):
                pred_img = (I_0_pred[i].clamp(-1, 1) + 1) / 2
                gt_img = (I_0[i].clamp(-1, 1) + 1) / 2
                
                name_prefix = f"sample_{batch_idx * batch_size + i}"
                
                save_image(pred_img, os.path.join(output_dir, f"{name_prefix}_generated.png"))
                
                save_image(gt_img, os.path.join(output_dir, f"{name_prefix}_original.png"))
                
                save_image((I_minus1[i].clamp(-1, 1) + 1) / 2, os.path.join(output_dir, f"{name_prefix}_frame_minus1.png"))
                save_image((I_plus1[i].clamp(-1, 1) + 1) / 2, os.path.join(output_dir, f"{name_prefix}_frame_plus1.png"))
                
                comparison = torch.cat([pred_img, gt_img], dim=2)  
                save_image(comparison, os.path.join(output_dir, f"{name_prefix}_comparison.png"))
    

if __name__ == "__main__":
    test_madiff()