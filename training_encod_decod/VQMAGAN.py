import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from .architecture import Encoder, EncoderWithFeatures, Decoder, PatchDiscriminator, device
import sys
import os.path as osp
# sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
# sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '.')))

class VQMAGAN(nn.Module):
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



# def calculate_perceptual_loss(pred, target):
#     return F.mse_loss(pred, target)



