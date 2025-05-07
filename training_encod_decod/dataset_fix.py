import os
from torch.utils.data import Dataset
from PIL import Image


class FrameTripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
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
    