import numpy as np
from skimage.metrics import structural_similarity as ssim
import lpips
import numpy as np
import torchvision.transforms as T



def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0 
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def ssin(img1, img2):
    score, _ = ssim(img1, img2, full=True, multichannel=True)  # або channel_axis=-1 для новіших версій


def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)


def mae(img1, img2):
    return np.mean(np.abs(img1 - img2))

def lpips_calc(img1, img2):
    transform = T.Compose([
        # T.Resize((H, W)),  # якщо потрібно змінити розмір
        T.ToTensor(),  # → [0,1]
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # → [-1,1]
    ])

    img1_tensor = transform(img1).unsqueeze(0)  # (1, 3, H, W)
    img2_tensor = transform(img2).unsqueeze(0)

    loss_fn = lpips.LPIPS(net='alex') 
    score = loss_fn(img1_tensor, img2_tensor)

    return score.item()




