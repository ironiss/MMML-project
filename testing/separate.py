import os
import shutil

mixed_dir = 'generated'
original_dir = 'originals'
os.makedirs(original_dir, exist_ok=True)

for filename in os.listdir(mixed_dir):
    if "original" in filename.lower(): 
        src = os.path.join(mixed_dir, filename)
        dst = os.path.join(original_dir, filename)
        shutil.copy(src, dst)
