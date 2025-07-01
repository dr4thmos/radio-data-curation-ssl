import os
from datetime import datetime
from utils import format_readable_run_name, compute_config_hash, log_on_mlflow
from parsers.cutouts_parser import get_parser_sliding_window
from data.lotss import LoTTSCollection
from astropy.io import fits
import numpy as np
import json

MLFLOW_EXPERIMENT_NAME = "cutting_out"
MLFLOW_RUN_NAME = "cutting_out"
ROOT_FOLDER = "outputs/cutouts"
OUTPUT_FOLDER_BASENAME = "cutouts"
METADATA_FILE_BASENAME = "metadata.json"
CUTOUTS_INFO_FILENAME = "info.json"

args = get_parser_sliding_window()
metadata = vars(args)
metadata["strategy"] = "sliding_window"

config_hash = compute_config_hash(metadata, exclude_keys=["mosaics_path"])
metadata["timestamp"] = datetime.now().isoformat()
metadata["config_hash"] = config_hash
metadata["root_folder"] = ROOT_FOLDER
metadata["cutouts_info_filename"] = CUTOUTS_INFO_FILENAME

readable_name_params = {}
readable_name_params["strategy"] = metadata["strategy"]
readable_name_params["overlap"] = metadata["overlap"]
readable_name_params["size"] = metadata["window_size"]
metadata["run_folder"] = format_readable_run_name(OUTPUT_FOLDER_BASENAME, readable_name_params, config_hash)

""" Logica Principale """

global_json_path = os.path.join(ROOT_FOLDER, metadata["run_folder"], CUTOUTS_INFO_FILENAME)

if os.path.exists(global_json_path):
    exit()

run_path = os.path.join(ROOT_FOLDER, metadata["run_folder"])
metadata_file_path = os.path.join(run_path, METADATA_FILE_BASENAME)
print(run_path)
if not os.path.exists(run_path):
    os.makedirs(run_path)
else:
    print(f"Esperimento già presente {run_path}")

window_size = args.window_size
overlap = 0.50
step = int(window_size * (1 - overlap))

# Collezione LoTSS
collection = LoTTSCollection(name="LoTTS", path=os.path.abspath(args.mosaics_path))
print(collection)

# Itera sui mosaici nella collezione
for mosaic in collection:
    cutout_info = []
    
    print(f"Mosaic: {mosaic.mosaic_name}, Path: {mosaic.mosaic_path}")
    image_path = mosaic.mosaic_path
    mosaic_data = fits.getdata(image_path)
    
    output_dir = os.path.join(ROOT_FOLDER, metadata["run_folder"], mosaic.mosaic_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "npy"), exist_ok=True)
    
    json_path = os.path.join(output_dir, CUTOUTS_INFO_FILENAME)
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
                except Exception as e:
                    print(f"Errore nel salvataggio del file {npy_path}: {e}")
    
    # Salva il JSON per il singolo mosaico
    with open(json_path, 'w') as json_file:
        json.dump(cutout_info, json_file, indent=4)

# Creazione dell'info.json globale basandosi sui file locali
global_output = {}
global_counter = 0
for mosaic in collection:
    json_path = os.path.join(ROOT_FOLDER, metadata["run_folder"], mosaic.mosaic_name, CUTOUTS_INFO_FILENAME)
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            mosaic_data = json.load(json_file)
            for entry in mosaic_data:
                global_output[str(global_counter)] = entry
                global_counter += 1

# Salva il JSON globale
global_json_path = os.path.join(ROOT_FOLDER, metadata["run_folder"], CUTOUTS_INFO_FILENAME)
with open(global_json_path, 'w') as f:
    json.dump(global_output, f, indent=4)

# Rimuove i file info.json locali solo se il salvataggio del JSON globale è andato a buon fine
if os.path.exists(global_json_path):
    for mosaic in collection:
        json_path = os.path.join(ROOT_FOLDER, metadata["run_folder"], mosaic.mosaic_name, "info.json")
        if os.path.exists(json_path):
            os.remove(json_path)

print(f"Generazione completata. JSON globale salvato in {global_json_path}")
metadata["num_cutouts"] = global_counter

log_on_mlflow(MLFLOW_EXPERIMENT_NAME, MLFLOW_RUN_NAME, metadata, metadata_file_path)
print(f"Esperimento {metadata['run_folder']} completato con successo.")