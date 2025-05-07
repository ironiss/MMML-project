import numpy as np
from skimage.metrics import structural_similarity as ssim
import lpips
import os
import numpy as np
import torchvision.transforms as T
from torch_fidelity import calculate_metrics
import pandas as pd
from PIL import Image

all_images_dir = 'generated'
resized_gen_dir = 'resized_generated'
os.makedirs(resized_gen_dir, exist_ok=True)

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0 
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def ssim(img1, img2):
    score, _ = ssim(img1, img2, full=True, multichannel=True) 
    return score


def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)


def mae(img1, img2):
    return np.mean(np.abs(img1 - img2))

def lpips_calc(img1, img2):
    transform = T.Compose([
        T.ToTensor(), 
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])

    img1_tensor = transform(img1).unsqueeze(0) 
    img2_tensor = transform(img2).unsqueeze(0)

    loss_fn = lpips.LPIPS(net='alex') 
    score = loss_fn(img1_tensor, img2_tensor)

    return score.item()


def lolpips(img1, img2, patch_size=64, stride=32):
    h, w, _ = img1.shape
    loss_fn = lpips.LPIPS(net='alex')

    transform = T.Compose([
        T.ToTensor(),  
        T.Normalize(mean=[0.5]*3, std=[0.5]*3) 
    ])

    scores = []
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch1 = img1[i:i+patch_size, j:j+patch_size]
            patch2 = img2[i:i+patch_size, j:j+patch_size]

            t1 = transform(patch1).unsqueeze(0)
            t2 = transform(patch2).unsqueeze(0)

            score = loss_fn(t1, t2).item()
            scores.append(score)

    return np.mean(scores)


def calculate_fid(real_dir, gen_dir):
    metrics = calculate_metrics(
        input1=real_dir,
        input2=gen_dir,
        cuda=True,
        isc=False,
        fid=True,
        kid=False
    )
    return metrics['frechet_inception_distance']




results = []

for fname in os.listdir(all_images_dir):
    if not fname.endswith('_original.png'):
        continue

    idx = fname.split('_')[1]
    orig_path = os.path.join(all_images_dir, fname)
    gen_path = os.path.join(all_images_dir, f'sample_{idx}_generated.png')

    if not os.path.exists(gen_path):
        continue

    orig_img = Image.open(orig_path).convert('RGB')
    gen_img = Image.open(gen_path).convert('RGB').resize(orig_img.size)

    resized_path = os.path.join(resized_gen_dir, f'sample_{idx}_generated.png')
    gen_img.save(resized_path)

    orig_np = np.array(orig_img).astype(np.float32)
    gen_np = np.array(gen_img).astype(np.float32)

    psnr_val = psnr(orig_np, gen_np)
    ssim_val = ssim(orig_np, gen_np)
    mse_val = mse(orig_np, gen_np)
    mae_val = mae(orig_np, gen_np)
    lpips_val = lpips_calc(orig_img, gen_img)
    lolpips_val = lolpips(orig_np, gen_np)

    results.append({
        'sample': idx,
        'psnr': psnr_val,
        'ssim': ssim_val,
        'mse': mse_val,
        'mae': mae_val,
        'lpips': lpips_val,
        'lolpips': lolpips_val
    })


df = pd.DataFrame(results)
print(df)
df.to_csv('metric_results.csv', index=False)