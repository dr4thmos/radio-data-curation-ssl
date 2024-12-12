import argparse
import os
import json

from data.lotss import LoTTSCollection
from astropy.io import fits
import numpy as np
from scipy.ndimage import label, find_objects

# Funzione per gestire gli argomenti della linea di comando
def parse_args():
    parser = argparse.ArgumentParser(description="Crea ritagli da mosaici LoTSS usando bounding boxes.")
    return parser.parse_args()

# Parsing degli argomenti
args = parse_args()

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
    print("Numero di regioni trovate:", num_features)
    bounding_boxes = find_objects(labeled_mask)
    print("Bounding boxes trovate:", len(bounding_boxes))
    return bounding_boxes

# Collezione LoTSS
collection = LoTTSCollection(name="LoTTS", path="/home/tcecconello/radioimgs/radio-data-curation-ssl/LoTSS")
print(collection)

# Itera sui mosaici nella collezione
for mosaic in collection:
    image_path = mosaic.mosaic_path
    mask_path = mosaic.mask_path

    mosaic_data = fits.getdata(image_path)
    mask = np.squeeze(fits.getdata(mask_path))

    bounding_boxes = find_bounding_boxes(mask)

    # Dizionario per salvare le informazioni sui cutout per il mosaico corrente
    mosaic_cutout_info = {"1.5": [], "128": [], "256": []}

    for idx, bbox in enumerate(bounding_boxes):
        print(f"Processing bounding box {idx}: {bbox}")

        if len(bbox) < 2:
            print(f"Skipping invalid bounding box {idx}: {bbox}")
            continue

        # Estrarre i limiti della bounding box
        x_min, x_max = bbox[0].start, bbox[0].stop
        y_min, y_max = bbox[1].start, bbox[1].stop

        # Calcolare la dimensione della bounding box
        size_x = x_max - x_min
        size_y = y_max - y_min
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        original_size = max(size_x, size_y)

        print(f"Original bounding box size: {size_x}x{size_y}, Center: ({center_x}, {center_y})")

        # Determinare le dimensioni dei cutout in base alla dimensione della sorgente
        cutout_sizes = []
        if original_size < 96:
            cutout_sizes = [int(1.5 * original_size), 128, 256]
        elif 96 <= original_size <= 192:
            cutout_sizes = [int(1.5 * original_size), 256]
        else:
            cutout_sizes = [256]

        print(f"Cutout sizes for bounding box {idx}: {cutout_sizes}")

        for cutout_size in cutout_sizes:
            half_size = cutout_size // 2

            # Calcolare i limiti del ritaglio
            new_x_min = max(center_x - half_size, 0)
            new_x_max = min(center_x + half_size, mosaic_data.shape[0])
            new_y_min = max(center_y - half_size, 0)
            new_y_max = min(center_y + half_size, mosaic_data.shape[1])

            #print(f"Patch size: {cutout_size}, Limits: X({new_x_min}:{new_x_max}), Y({new_y_min}:{new_y_max})")

            # Estrarre il ritaglio
            patch = mosaic_data[new_x_min:new_x_max, new_y_min:new_y_max]

            # Verifica validità del ritaglio
            if patch.size == 0:
                print(f"Skipping empty patch for bounding box {idx} with size {cutout_size}")
                continue

            if np.isnan(patch).any():
                print(f"Skipping patch with NaN values for bounding box {idx}")
                continue

            # Determinare la directory di destinazione
            folder_key = "1.5" if cutout_size == int(1.5 * original_size) else str(cutout_size)
            output_dir = os.path.join(output_base_folders[folder_key], mosaic.mosaic_name)
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, "npy"), exist_ok=True)

            # Salva il ritaglio
            npy_filename = f"patch_{idx}_size_{cutout_size}.npy"
            npy_path = os.path.join(output_dir, "npy", npy_filename)

            np.save(npy_path, patch)
            print(f"Saved patch: {npy_path}")

            # Aggiungere informazioni al file JSON per il mosaico corrente
            mosaic_cutout_info[folder_key].append({
                "mosaic": mosaic.mosaic_name,
                "filename": npy_filename,
                "position": [center_x, center_y],
                "size": cutout_size
            })

    # Salva le informazioni su file JSON per il mosaico corrente
    for folder_key, info in mosaic_cutout_info.items():
        mosaic_folder = os.path.join(output_base_folders[folder_key], mosaic.mosaic_name)
        json_path = os.path.join(mosaic_folder, f"info_mc_cutouts_{folder_key}.json")
        os.makedirs(mosaic_folder, exist_ok=True)
        with open(json_path, 'w') as json_file:
            json.dump(info, json_file, indent=4)
            print(f"Saved JSON for {folder_key} in {json_path}")
