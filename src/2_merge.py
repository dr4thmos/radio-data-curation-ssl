import os
from datetime import datetime
from utils import format_readable_run_name, compute_config_hash, log_on_mlflow
from parsers.merge_parser import get_parser
import mlflow
import json

MLFLOW_EXPERIMENT_NAME = "merge_cutouts"
MLFLOW_RUN_NAME = "merge_cutouts"
ROOT_FOLDER = "outputs/merged_cutouts"
OUTPUT_FOLDER_BASENAME = "merged_cutouts"
METADATA_FILE_BASENAME = "metadata.json"
MERGED_CUTOUTS_INFO_FILENAME = "info.json"

"""
compute hash based on configs/args
add info to metadata (root folder could be moved into args with default parameter?)
"""
args = get_parser()
metadata = vars(args)
config_hash = compute_config_hash(metadata)
metadata["timestamp"] = datetime.now().isoformat()
metadata["config_hash"] = config_hash
metadata["root_folder"] = ROOT_FOLDER
metadata["merged_cutouts_info_filename"] = MERGED_CUTOUTS_INFO_FILENAME

readable_name_params = {}
metadata["run_folder"] = format_readable_run_name(OUTPUT_FOLDER_BASENAME, readable_name_params, config_hash)

""" Logica Principale """

run_path = os.path.join(ROOT_FOLDER, metadata["run_folder"])
metadata_file_path = os.path.join(run_path, METADATA_FILE_BASENAME)
print(run_path)
if not os.path.exists(run_path):
    os.makedirs(run_path)
else:
    print(f"Esperimento già presente per ricomputarlo eliminare la cartella {run_path}")
    exit()

"""
# Recupera dai cutouts_id su mlflow i path delle liste dei cutouts e uniscili
# I cutout_id sono già in metadata
cutout_paths = []
for cutout_id in metadata.get("cutout_ids", []):
    print(cutout_id)
    try:
        run = mlflow.get_run(cutout_id)
        cutout_path = run.data.params.get("cutout_path")
        if cutout_path:
            cutout_paths.append(cutout_path)
    except mlflow.exceptions.MlflowException as e:
        print(f"Errore nel recuperare il cutout_id {cutout_id}: {e}")
"""
# Leggiamo i json e salviamoli nel formato "index": {...info...} aggiungiamo come informazione il cutout_ids da cui proviene
cutout_paths = []
for cutout_id in metadata.get("cutout_ids", []):
    print(f"Recupero cutout_id: {cutout_id}")
    try:
        run = mlflow.get_run(cutout_id)
        cutouts_root_folder = run.data.params.get("root_folder")
        cutouts_run_folder = run.data.params.get("run_folder")
        cutouts_info_filename = run.data.params.get("cutouts_info_filename")
        cutout_path = os.path.join(cutouts_root_folder, cutouts_run_folder, cutouts_info_filename)
        if cutout_path:
            cutout_paths.append(cutout_path)
    except mlflow.exceptions.MlflowException as e:
        print(f"Errore nel recuperare il cutout_id {cutout_id}: {e}")

# Unione dei JSON
merged_output = {}
global_index = 0
for cutout_path in cutout_paths:
    if os.path.exists(cutout_path):
        try:
            with open(cutout_path, 'r') as json_file:
                cutout_data = json.load(json_file)
                for entry in cutout_data.values():
                    merged_output[str(global_index)] = entry
                    global_index += 1
        except Exception as e:
            print(f"Errore nel leggere il file {cutout_path}: {e}")
    else:
        print(f"Percorso non trovato: {cutout_path}")

# Salvataggio del JSON unito
global_json_path = os.path.join(run_path, MERGED_CUTOUTS_INFO_FILENAME)
with open(global_json_path, 'w') as f:
    json.dump(merged_output, f, indent=4)

print(f"Unione completata. JSON globale salvato in {global_json_path}")

metadata["num_cutouts"] = global_index

# check if folder is present -> exit -> else metadata["version"] = v1
# with parameter update, overwrite it, add version +1

log_on_mlflow(MLFLOW_EXPERIMENT_NAME, MLFLOW_RUN_NAME, metadata, metadata_file_path)
print(f"Esperimento {metadata['run_folder']} completato con successo.")


