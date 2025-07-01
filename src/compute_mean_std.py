import numpy as np
import os
import tqdm
from data.lotss import LoTTSCollection
from astropy.io import fits

def compute_mean_std_numpy(dataset):
    sum_pixels = 0.0
    sum_sq_pixels = 0.0
    num_pixels = 0
    counter = 0
    for mosaic in dataset:
        print(counter)
        counter += 1
        mosaic_path = mosaic.mosaic_path
        img = fits.getdata(mosaic_path)
        img = img.astype(np.float32)
        valid_pixels = img[~np.isnan(img)]  # Ignore NaNs
        sum_pixels += np.sum(valid_pixels)
        sum_sq_pixels += np.sum(valid_pixels ** 2)
        num_pixels += valid_pixels.size

    mean = sum_pixels / num_pixels
    std = np.sqrt((sum_sq_pixels / num_pixels) - (mean ** 2))
    return mean, std

collection = LoTTSCollection(name="LoTTS", path="/leonardo_scratch/fast/INA24_C5B09/LoTSS/DR2")
mean, std = compute_mean_std_numpy(collection)
print("Mean:", mean)
print("Std:", std)
