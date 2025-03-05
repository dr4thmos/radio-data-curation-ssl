import argparse
import os
import json

from data.lotss import LoTTSCollection
from astropy.io import fits
import numpy as np
from scipy.ndimage import label, find_objects

# Percorso base per i cutout
output_base_folders = {
    "1.5": "/home/tcecconello/radioimgs/radio-data-curation-ssl/LoTSS/cutouts/mc_cutouts_1.5",
    "128": "/home/tcecconello/radioimgs/radio-data-curation-ssl/LoTSS/cutouts/mc_cutouts_128",
    "256": "/home/tcecconello/radioimgs/radio-data-curation-ssl/LoTSS/cutouts/mc_cutouts_256"
}
for folder in output_base_folders.values():
    os.makedirs(folder, exist_ok=True)

# Funzione per trovare le bounding boxes
def find_bounding_boxes(mask):
    """Trova le bounding box di tutte le maschere binarie (regioni con valore 1)."""
    if mask.dtype.byteorder == '>':
        mask = mask.byteswap().view(mask.dtype.newbyteorder('<'))
    labeled_mask, num_features = label(mask)
    bounding_boxes = find_objects(labeled_mask)
    return bounding_boxes

# Collezione LoTSS
collection = LoTTSCollection(name="LoTTS", path="/home/tcecconello/radioimgs/radio-data-curation-ssl/LoTSS")

# Dizionario globale per salvare tutte le informazioni
all_cutout_info = {}
global_counter = 0

# Itera sui mosaici nella collezione
for mosaic in collection:
    image_path = mosaic.mosaic_path
    mask_path = mosaic.mask_path
    mosaic_data = fits.getdata(image_path)
    mask = np.squeeze(fits.getdata(mask_path))
    bounding_boxes = find_bounding_boxes(mask)
    mosaic_cutout_info = {"1.5": [], "128": [], "256": []}

    for idx, bbox in enumerate(bounding_boxes):
        if len(bbox) < 2:
            continue
        x_min, x_max = bbox[0].start, bbox[0].stop
        y_min, y_max = bbox[1].start, bbox[1].stop
        size_x = x_max - x_min
        size_y = y_max - y_min
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        original_size = max(size_x, size_y)

        cutout_sizes = []
        if original_size < 96:
            cutout_sizes = [int(1.5 * original_size), 128, 256]
        elif 96 <= original_size <= 192:
            cutout_sizes = [int(1.5 * original_size), 256]
        else:
            cutout_sizes = [256]

        for cutout_size in cutout_sizes:
            half_size = cutout_size // 2
            new_x_min = max(center_x - half_size, 0)
            new_x_max = min(center_x + half_size, mosaic_data.shape[0])
            new_y_min = max(center_y - half_size, 0)
            new_y_max = min(center_y + half_size, mosaic_data.shape[1])
            patch = mosaic_data[new_x_min:new_x_max, new_y_min:new_y_max]

            if patch.size == 0 or np.isnan(patch).any():
                continue

            folder_key = "1.5" if cutout_size == int(1.5 * original_size) else str(cutout_size)
            output_dir = os.path.join(output_base_folders[folder_key], mosaic.mosaic_name)
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, "npy"), exist_ok=True)
            npy_filename = f"patch_{idx}_size_{cutout_size}.npy"
            npy_path = os.path.join(output_dir, "npy", npy_filename)

            try:
                np.save(npy_path, patch)
                cutout_entry = {
                    "mosaic_name": mosaic.mosaic_name,
                    "survey": collection.name,
                    "filename": npy_filename,
                    "position": [center_x, center_y],
                    "size": cutout_size,
                    "file_path": npy_path
                }
                mosaic_cutout_info[folder_key].append(cutout_entry)
                all_cutout_info[str(global_counter)] = cutout_entry
                global_counter += 1
            except:
                pass

    for folder_key, info in mosaic_cutout_info.items():
        mosaic_folder = os.path.join(output_base_folders[folder_key], mosaic.mosaic_name)
        json_path = os.path.join(mosaic_folder, "info.json")
        with open(json_path, 'w') as json_file:
            json.dump(info, json_file, indent=4)

# Scrive il file JSON globale
global_json_path = os.path.join("/home/tcecconello/radioimgs/radio-data-curation-ssl/LoTSS/cutouts", "info.json")
with open(global_json_path, 'w') as global_json_file:
    json.dump(all_cutout_info, global_json_file, indent=4)
