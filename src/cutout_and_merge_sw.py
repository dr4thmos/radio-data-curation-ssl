import argparse
import os
import json
from data.lotss import LoTTSCollection
from astropy.io import fits
import numpy as np

# Funzione per gestire gli argomenti della linea di comando
def parse_args():
    parser = argparse.ArgumentParser(description="Crea ritagli da mosaici LoTSS e genera il JSON riassuntivo.")
    parser.add_argument("--window_size", type=int, default=256, help="Dimensione della finestra per i ritagli.")
    return parser.parse_args()

# Parsing degli argomenti
args = parse_args()
window_size = args.window_size
overlap = 0.50
step = int(window_size * (1 - overlap))

# Percorso base per i cutout
cutout_root = f"/home/tcecconello/radioimgs/radio-data-curation-ssl/LoTSS/cutouts/sw_cutouts_{window_size}"
os.makedirs(cutout_root, exist_ok=True)

# Collezione LoTSS
collection = LoTTSCollection(name="LoTTS", path="/home/tcecconello/radioimgs/radio-data-curation-ssl/LoTSS")
print(collection)

# Dizionario per il file JSON globale
global_output = {}
global_counter = 0

# Itera sui mosaici nella collezione
for mosaic in collection:
    cutout_info = []
    
    print(f"Mosaic: {mosaic.mosaic_name}, Path: {mosaic.mosaic_path}")
    image_path = mosaic.mosaic_path
    mosaic_data = fits.getdata(image_path)
    
    output_dir = os.path.join(cutout_root, mosaic.mosaic_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "npy"), exist_ok=True)
    
    json_path = os.path.join(output_dir, "info.json")
    if os.path.exists(json_path):
        print(f"Saltato {mosaic.mosaic_name} perché info.json esiste già.")
        continue
    
    for i in range(0, mosaic_data.shape[0] - window_size + 1, step):
        for j in range(0, mosaic_data.shape[1] - window_size + 1, step):
            patch = mosaic_data[i:i + window_size, j:j + window_size]
            
            if not np.isnan(patch).any():
                npy_filename = f"patch_{i}_{j}.npy"
                npy_path = os.path.join(output_dir, "npy", npy_filename)
                
                try:
                    np.save(npy_path, patch)
                    if os.path.exists(npy_path):
                        cutout_metadata = {
                            "file_path": npy_path,
                            "survey": collection.name,
                            "mosaic_name": mosaic.mosaic_name,
                            "position": [i, j],
                            "size": window_size
                        }
                        cutout_info.append(cutout_metadata)
                        global_output[str(global_counter)] = cutout_metadata
                        global_counter += 1
                except Exception as e:
                    print(f"Errore nel salvataggio del file {npy_path}: {e}")
    
    # Salva il JSON per il singolo mosaico
    with open(json_path, 'w') as json_file:
        json.dump(cutout_info, json_file, indent=4)

# Salva il JSON globale con tutti i metadati
global_json_path = os.path.join(cutout_root, "info.json")
with open(global_json_path, 'w') as f:
    json.dump(global_output, f, indent=4)

print(f"Generazione completata. JSON globale salvato in {global_json_path}")
