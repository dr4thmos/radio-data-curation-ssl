import argparse
import os
import json

from data.lotss import LoTTSCollection
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# Funzione per gestire gli argomenti della linea di comando
def parse_args():
    parser = argparse.ArgumentParser(description="Crea ritagli da mosaici LoTSS usando una finestra scorrevole.")
    parser.add_argument("--window_size", type=int, default=256, help="Dimensione della finestra per i ritagli.")
    return parser.parse_args()

# Parsing degli argomenti
args = parse_args()
window_size = args.window_size
overlap = 0.50
step = int(window_size * (1 - overlap))

# Percorso base per i cutout
output_base_folder = f"/home/tcecconello/radioimgs/radio-data-curation-ssl/LoTSS/cutouts/sw_cutouts_{window_size}"
if not os.path.exists(output_base_folder):
    os.makedirs(output_base_folder)

# Collezione LoTSS
collection = LoTTSCollection(name="LoTTS", path="/home/tcecconello/radioimgs/radio-data-curation-ssl/LoTSS")
print(collection)

# Dizionario per salvare le informazioni sui cutout
cutout_info = []

# Itera sui mosaici nella collezione
for mosaic in collection:
    print(f"Mosaic: {mosaic.mosaic_name}, Path: {mosaic.mosaic_path}, Mask: {mosaic.mask_path}")
    
    image_path = mosaic.mosaic_path
    mask_path = mosaic.mask_path
    
    mosaic_data = fits.getdata(image_path)
    mask = np.squeeze(fits.getdata(mask_path))

    output_dir = os.path.join(output_base_folder, mosaic.mosaic_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        continue
    if not os.path.exists(os.path.join(output_dir, "npy")):
        os.makedirs(os.path.join(output_dir, "npy"))
    #if not os.path.exists(os.path.join(output_dir, "previews")):
    #    os.makedirs(os.path.join(output_dir, "previews"))

    for i in range(0, mosaic_data.shape[0] - window_size + 1, step):
        for j in range(0, mosaic_data.shape[1] - window_size + 1, step):
            patch = mosaic_data[i:i + window_size, j:j + window_size]
            
            if not np.isnan(patch).any():
                npy_filename = f"patch_{i}_{j}.npy"
                #png_filename = f"patch_{i}_{j}.png"

                npy_path = os.path.join(output_dir, "npy", npy_filename)
                #png_path = os.path.join(output_dir, "previews", png_filename)

                np.save(npy_path, patch)
                #plt.imsave(png_path, patch, cmap='gray')

                cutout_info.append({
                    "filename": npy_filename,
                    "survey": collection.name,
                    "mosaic_name": mosaic.mosaic_name,
                    "position": [i, j],
                    "size": window_size
                    # aggiungere informazioni sui vicini
                })

    # Salva le informazioni su file JSON
    json_path = os.path.join(output_dir, f"info_sw_{window_size}.json")
    with open(json_path, 'w') as json_file:
        json.dump(cutout_info, json_file, indent=4)
