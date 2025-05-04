import os
import torch
import argparse
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import sys
from argparse import Namespace

# Add parent directory to path to import project modules
import os.path as osp
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))

# Import project modules
from training_encod_decod.VQMAGAN import VQMAGAN
from EVENTGAN.models.eventgan_trainer import EventGANTrainer
from EVENTGAN.pytorch_utils.base_options import BaseOptions
import configs
from sampling.maddif import MADiff  # Adjust this import path if needed


def load_image(path, device):
    """Load and preprocess an image for the model."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0).to(device)


def process_sequence(prev_frame_path, next_frame_path, output_path, madiff, eventgan, device, steps=100):
    """Process a single sequence of frames."""
    # Load frames
    I_prev = load_image(prev_frame_path, device)
    I_next = load_image(next_frame_path, device)
    
    # Generate intermediate frame
    with torch.no_grad():
        interpolated = madiff.sample(I_prev, I_next, eventgan, steps=steps)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the interpolated frame
    save_image(interpolated[0].clamp(-1, 1), output_path)
    print(f"[✓] Saved interpolated frame to {output_path}")
    
    return interpolated


def process_folder(input_folder, output_folder, madiff, eventgan, device, steps=100):
    """Process all valid frame pairs in a folder."""
    if not os.path.exists(input_folder):
        print(f"[!] Input folder {input_folder} does not exist")
        return
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files in the folder
    all_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(all_files) < 2:
        print(f"[!] Not enough frames in {input_folder} (found {len(all_files)})")
        return
    
    # Process all consecutive pairs
    for i in range(len(all_files) - 1):
        prev_frame_path = os.path.join(input_folder, all_files[i])
        next_frame_path = os.path.join(input_folder, all_files[i+1])
        output_path = os.path.join(output_folder, f"interpolated_{i}_to_{i+1}.png")
        
        process_sequence(prev_frame_path, next_frame_path, output_path, madiff, eventgan, device, steps)


def process_dataset(input_root, output_root, madiff, eventgan, device, steps=100):
    """Process a dataset with multiple sequences."""
    i = 0
    for set_folder in os.listdir(input_root):
        set_path = os.path.join(input_root, set_folder)
        if not os.path.isdir(set_path): 
            continue

        for sequence_folder in os.listdir(set_path):
            sequence_path = os.path.join(set_path, sequence_folder)
            if not os.path.isdir(sequence_path): 
                continue

            # Check for required frames (assuming standard naming)
            frame0 = os.path.join(sequence_path, 'im1.png')
            frame2 = os.path.join(sequence_path, 'im3.png')

            if not (os.path.exists(frame0) and os.path.exists(frame2)):
                print(f"[!] Skipping {sequence_path} - missing required frames")
                continue

            # Create output folder
            # output_path = os.path.join(output_root, f'frame{i}_pred.png')
            # i+=1
            output_filename = f"{set_folder}_{sequence_folder}.png"
            output_path = os.path.join(output_root, output_filename)

            # Process the sequence
            # process_sequence(frame0, frame2, output_path, madiff, eventgan, device, steps)
            try:
                process_sequence(frame0, frame2, output_path, madiff, eventgan, device, steps)
            except Exception as e:
                print(f"[!] Error processing {sequence_folder}: {e}")


def get_args(parser):
    parser.add_argument('--name', type=str)
    parser.add_argument('--model', type=str)

def main():
    # parser = argparse.ArgumentParser(description='Test MADiff model for frame interpolation')
    # parser.add_argument('--vq_magan_weights', type=str, required=True, help='Path to VQ-MAGAN weights')
    # parser.add_argument('--madiff_weights', type=str, required=True, help='Path to MADiff weights')
    # parser.add_argument('--eventgan_weights', type=str, required=True, help='Path to EventGAN weights')
    # parser.add_argument('--input', type=str, required=True, help='Path to input folder or dataset')
    # parser.add_argument('--output', type=str, default='results', help='Path to output folder')
    # parser.add_argument('--steps', type=int, default=100, help='Number of diffusion steps')
    # parser.add_argument('--mode', type=str, default='dataset', choices=['single', 'folder', 'dataset'], 
    #                     help='Processing mode: single pair, folder of images, or dataset')
    # parser.add_argument('--frame1', type=str, help='Path to first frame (only for single mode)')
    # parser.add_argument('--frame2', type=str, help='Path to second frame (only for single mode)')
    # # args = parser.parse_args()
    # parser.add_argument('--name', type=str, default='EventGAN')
    # parser.add_argument('--model', type=str, default='EventGAN')

    # args = parser.parse_args()


    # # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"[✓] Using device: {device}")

    # # Load VQ-MAGAN model
    # print("[*] Loading VQ-MAGAN model...")
    vq_magan = VQMAGAN(in_channels=3, hidden_dim=64).to(device)
    
    try:
        checkpoint = torch.load("/home/ironiss/Documents/mmml-project/MMML-project/MMML-project-Develop/sampling/vq_magan_checkpoint_epoch20.pt", map_location=device)
        
        # Check if checkpoint has expected structure
        if isinstance(checkpoint, dict) and 'encoder_state_dict' in checkpoint:
            vq_magan.encoder.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            vq_magan.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            vq_magan.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            print("[✓] VQ-MAGAN loaded successfully (checkpoint format)")
        else:
            # Maybe checkpoint is already a state_dict
            vq_magan.load_state_dict(checkpoint, strict=False)
            print("[✓] VQ-MAGAN loaded successfully (direct format)")
    except Exception as e:
        print(f"[!] Error loading VQ-MAGAN: {e}")
        print("[!] Continuing with partially loaded model")
    
    vq_magan.eval()

    # # Load MADiff model
    print("[*] Loading MADiff model...")
    madiff = MADiff(vq_magan=vq_magan).to(device)
    
    try:
        madiff_checkpoint = torch.load("/home/ironiss/Downloads/madiff_checkpoint_epoch40.pt", map_location=device)
        if 'denoising_unet_state_dict' in madiff_checkpoint:
            madiff.denoising_unet.load_state_dict(madiff_checkpoint['denoising_unet_state_dict'])
            print("[✓] MADiff loaded successfully (checkpoint format)")
        else:
            madiff.load_state_dict(madiff_checkpoint)
            print("[✓] MADiff loaded successfully (direct format)")
    except Exception as e:
        print(f"[!] Error loading MADiff: {e}")
    
    madiff.eval()

    print("[*] Loading EventGAN model...")

    options = BaseOptions()
    options.parser = configs.get_args(options.parser)
    args = options.parse_args()
    eventgan_trainer = EventGANTrainer(options=args, train=False)
    eventgan_checkpoint = torch.load("/home/ironiss/Documents/mmml-project/MMML-project/MMML-project-Develop/sampling/Weights.pt", map_location=device)
    eventgan_trainer.models_dict["gen"].load_state_dict(eventgan_checkpoint['gen'])
    eventgan_model = eventgan_trainer.models_dict["gen"].to(device)
    eventgan_model.eval()
    print("[✓] MADiff loaded successfully (checkpoint format)")
    # # Process based on selected mode
    
    process_dataset("/home/ironiss/Documents/mmml-project/MMML-project/MMML-project-Develop/testing/sequences", "/home/ironiss/Documents/mmml-project/MMML-project/MMML-project-Develop/testing/generated", madiff, eventgan_model, device)
    
    print("----------------------------------------")
    print("[✓] Processing completed!")


if __name__ == "__main__":
    main()