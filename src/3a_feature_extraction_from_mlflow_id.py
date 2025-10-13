# ──────────────────────────────────────────────────────────────────────────────
#   IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
from datetime import datetime
from utils import format_readable_run_name, compute_config_hash, log_on_mlflow
from parsers.extract_features_parser import get_parser

import sys
from tqdm import tqdm 
import os
import mlflow
import json
import numpy as np
import torchvision.transforms as T
import torch
import re
import traceback
import h5py

from transformers import (
    AutoConfig, AutoModel, AutoModelForImageClassification, AutoImageProcessor
)
from collections import OrderedDict
from thingsvision import get_extractor_from_model, get_extractor
from torchvision.models import resnet18, resnet50
from thingsvision.utils.data import DataLoader
from data.custom import CustomUnlabeledDatasetWithPath
from torch.utils.data import DataLoader as PyTorchDataLoader


class MinMaxNormalize(object):
    """Normalizza un tensore per istanza nel range [0, 1]."""
    def __call__(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val > 0:
            tensor = (tensor - min_val) / (max_val - min_val)
        else:
            tensor.fill_(0) 
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'



def parse_model_parameters(param_list):
    """Parse model parameters from a list of key=value strings."""
    return {kv.split('=')[0]: kv.split('=')[1] for kv in param_list if '=' in kv}

# ──────────────────────────────────────────────────────────────────────────────
#   CONSTANTS & SETUP
# ──────────────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "extract_features"
MLFLOW_RUN_NAME = "extract_features"
ROOT_FOLDER = "outputs/features"
OUTPUT_FOLDER_BASENAME = "features"
METADATA_FILE_BASENAME = "metadata.json"
# Il nostro nuovo file di output intelligente
FINAL_HDF5_FILENAME = "features_data_fast.h5"

# ──────────────────────────────────────────────────────────────────────────────
#   ARGUMENT PARSING & FOLDER SETUP
# ──────────────────────────────────────────────────────────────────────────────
args = get_parser()
metadata = vars(args)

# Create unique hash for config and setup folders
config_hash = compute_config_hash(metadata, exclude_keys=["cuda_devices"])
metadata["timestamp"] = datetime.now().isoformat()
metadata["config_hash"] = config_hash
metadata["root_folder"] = ROOT_FOLDER

# Create readable folder name
readable_name_params = {}
readable_name_params["model"] = metadata.get("model_name", "")
readable_name_params["variant"] = metadata.get("variant", "")
metadata["run_folder"] = format_readable_run_name(OUTPUT_FOLDER_BASENAME, readable_name_params, config_hash)
metadata["features_filename"] = FINAL_HDF5_FILENAME

# Create output directory
run_path = os.path.join(metadata["root_folder"], metadata["run_folder"])
metadata_file_path = os.path.join(run_path, METADATA_FILE_BASENAME)

print(f"Output directory: {run_path}")
if not os.path.exists(run_path):
    os.makedirs(run_path)
else:
    if os.listdir(run_path):
        print(f"Experiment already exists. To recompute, delete folder {run_path}")
        exit()
    else:
        print(f"Folder {run_path} exists but is empty, proceeding with experiment.")

# ──────────────────────────────────────────────────────────────────────────────
#   MLFLOW DATA RETRIEVAL
# ──────────────────────────────────────────────────────────────────────────────
try:
    run = mlflow.get_run(metadata["source_id"])
    merged_cutouts_root_folder = run.data.params.get("root_folder")
    merged_cutouts_run_folder = run.data.params.get("run_folder")
    merged_cutouts_info_filename = run.data.params.get("merged_cutouts_info_filename")
    merged_cutout_folder_path = os.path.join(merged_cutouts_root_folder, merged_cutouts_run_folder)
    merged_cutout_path = os.path.join(merged_cutouts_root_folder, merged_cutouts_run_folder, merged_cutouts_info_filename)
except mlflow.exceptions.MlflowException as e:
    print(f"Error retrieving cutout_id {metadata['source_id']}: {e}")
    exit(1)

# Load cutout information
with open(merged_cutout_path, "r") as f:
    merged_cutouts_info = json.load(f)
    n_cutouts = len(merged_cutouts_info)
    print(f"Found {n_cutouts} cutouts to process")

# ──────────────────────────────────────────────────────────────────────────────
#   MODEL, EXTRACTOR, AND TRANSFORM INITIALIZATION
# ──────────────────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize model based on type
model_parameters = parse_model_parameters(args.model_parameters)
if args.model_type == "cecconello_ssl":
    if not args.ckpt_path or not args.model_name:
        print("Errore: Per model_type 'cecconello_ssl', sono richiesti sia --model_name che --ckpt_path.")
        exit(1)
    model_metadata_path = os.path.join(os.path.dirname(args.ckpt_path), "args.json")
    
    with open(model_metadata_path, "r") as f:
        model_metadata = json.load(f)
    
    out_path = os.path.join(args.features_folder, model_metadata["name"])
    
    print("ckpt_path '{args.ckpt_path}")
    print("Modello '{args.model_name}")

    if args.model_name == "resnet18":
        model = resnet18(weights=None)
    elif args.model_name == "resnet50":
        model = resnet50(weights=None)
    else:
        print(f"Errore: Modello '{args.model_name}' non supportato per model_type 'cecconello_ssl'. Usare 'resnet18' o 'resnet50'.")
        exit(1)
    model.fc = torch.nn.Identity()
    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)

    backbone_state_dict = OrderedDict(
        [(k.replace("backbone.", ""), v) for k, v in checkpoint["state_dict"].items() if k.startswith("backbone.")]
    )

    model.load_state_dict(backbone_state_dict)
    backend = 'pt'

    extractor = get_extractor_from_model(
        model=model, 
        device=device,
        backend=backend
    )

USE_GRAYSCALE = True
if args.model_type == "andrea_dino":
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_channels=1 if USE_GRAYSCALE else 3,
    )
    model = AutoModel.from_pretrained(
        args.model_name, config=config, ignore_mismatched_sizes=USE_GRAYSCALE
    ).to(device)
    backend = 'pt'

    extractor = get_extractor_from_model(
        model=model, 
        device=device,
        backend=backend
    )
    print(extractor.show_model())

elif args.model_type == "thingsvision":
    variant_name = "-".join(str(value) for value in model_parameters.values())
    sanitized_variant = re.sub(r'[<>:"/\\|?*]', '_', variant_name)
    out_path = os.path.join(args.features_folder, args.model_name, sanitized_variant)

    extractor = get_extractor(
        model_name=args.model_name,
        source=args.source,
        device=device,
        pretrained=True,
        model_parameters=model_parameters
    )



# 1. Definiamo la pipeline di base.
# I tuoi dati sono sempre 1-canale e float, quindi ToTensor non scala il range.
transform_list = [
    T.ToTensor()
]

print(f"I dati di input sono a 1 canale.")
print(f"Il modello si aspetta un input a {args.model_input_channels} canali.")
print(f"Normalizzazione scelta: {args.normalization}")

# 2. Applichiamo la normalizzazione richiesta.
# Questa logica opera sui dati a 1 canale, prima di ogni manipolazione dei canali.
if args.normalization == "minmax":
    transform_list.append(MinMaxNormalize())
    print("-> Step 1: Applico normalizzazione MinMax per istanza (range [0, 1]).")
else:
    # Per tutte le altre normalizzazioni, usiamo valori a 1 canale.
    if args.normalization == "imagenet":
        mean, std = [0.45], [0.225]
    elif args.normalization == "mean05std05":
        mean, std = [0.5], [0.5]
    elif args.normalization == "mean05std0225":
        mean, std = [0.5], [0.225]
    elif args.normalization == "meanstdLoTSS":
        mean, std = [2e-05], [0.003]
    else:
        print(f"!! WARNING: Normalizzazione '{args.normalization}' non riconosciuta. Uso 'minmax' come fallback sicuro.")
        mean, std = None, None
        transform_list.append(MinMaxNormalize())
    
    if mean is not None:
        print(f"-> Step 1: Applico normalizzazione con media/std (mean={mean}, std={std}).")
        transform_list.append(T.Normalize(mean, std))

# 3. Adattiamo i canali dei dati ai requisiti del modello.
# Questo è il "ponte" tra i dati (1 ch) e il modello (N ch).
if args.model_input_channels == 3:
    print("-> Step 2: Duplico il canale 3 volte per adattarlo al modello.")
    transform_list.append(T.Lambda(lambda x: x.repeat(3, 1, 1)))
elif args.model_input_channels == 1:
    print("-> Step 2: Il modello accetta 1 canale, nessuna modifica necessaria.")
    pass # Esplicito che non facciamo nulla

# 4. Applichiamo il resize finale.
print(f"-> Step 3: Ridimensiono l'output a {args.resize}x{args.resize} pixel.")
transform_list.append(T.Resize(args.resize))

# 5. Componiamo la pipeline finale.
transforms = T.Compose(transform_list)

print("\n--- Pipeline di Trasformazioni Finale ---")
print(transforms)
print("-" * 40 + "\n")



# ──────────────────────────────────────────────────────────────────────────────
#   DATASET & DATALOADER SETUP
# ──────────────────────────────────────────────────────────────────────────────
print(f"Loading dataset from {merged_cutout_folder_path}")
dataset = CustomUnlabeledDatasetWithPath(
    data_path=merged_cutout_folder_path,
    loader_type="npy",
    transforms=transforms,
    datalist=merged_cutouts_info_filename
)
print(f"Dataset size: {len(dataset)}")

if args.test_mode:
    metadata["test_mode"] = True
    metadata["test_batches"] = args.test_batches
    print(f"TEST MODE ACTIVE: Limited to {args.test_batches} batches")
else:
    metadata["test_mode"] = False
    print("FULL MODE: Processing all batches")

NUM_WORKERS = min(os.cpu_count(), 8) if device == 'cuda' else 0
OPTIMIZED_BATCH_SIZE = args.batch_size
print(f"Optimizing DataLoader with: a batch_size of {OPTIMIZED_BATCH_SIZE}, {NUM_WORKERS} workers, pin_memory=True")


dataloader = PyTorchDataLoader(
    dataset=dataset,
    batch_size=OPTIMIZED_BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,          # Funziona solo se num_workers > 0
    prefetch_factor=2 if NUM_WORKERS > 0 else None, # Funziona solo se num_workers > 0
    shuffle=False,            # IMPORTANTE: non mescolare i dati per salvare correttamente i path!
    drop_last=False           # IMPORTANTE: processa anche l'ultimo batch se è più piccolo
)
"""
dataloader = DataLoader(
    dataset=dataset, 
    batch_size=OPTIMIZED_BATCH_SIZE, 
    backend=extractor.get_backend(),
    # Argomenti per l'ottimizzazione
    num_workers=NUM_WORKERS,
    pin_memory=True,          # Accelera il trasferimento dati a CUDA
    prefetch_factor=2         # Chiede ai worker di pre-caricare 2 batch per worker
)
"""

#dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, backend=extractor.get_backend())

# ──────────────────────────────────────────────────────────────────────────────
#   FEATURE EXTRACTION WITH DIRECT HDF5 WRITING
# ──────────────────────────────────────────────────────────────────────────────
print("\n--- Starting feature extraction with robust HDF5 writing ---")

# 1. Calcoliamo in anticipo le dimensioni finali
total_rows = len(dataset)
# --- START FIX: Spacchetta l'output del dataloader e usa solo le immagini ---
temp_batch_images, _ = next(iter(dataloader)) # Ignoriamo i percorsi con _

if args.model_type == "andrea_dino":
    # Passa solo il tensore delle immagini all'estrattore
    raw_features = extractor.extract_features(batches=[temp_batch_images], module_name=args.module_name, flatten_acts=False, output_type="ndarray")
    feature_dim = raw_features.shape[2] 
else:
    # Passa solo il tensore delle immagini all'estrattore
    raw_features = extractor.extract_features(batches=[temp_batch_images], module_name=args.module_name, flatten_acts=True, output_type="ndarray")
    feature_dim = raw_features.shape[1]
# --- END FIX ---
"""
# --- START FIX: Gestisce il nuovo output del dataloader (immagini, percorsi) ---
temp_batch_images, _ = next(iter(dataloader))
if args.model_type == "andrea_dino":
    raw_features = extractor.extract_features(batches=[temp_batch_images], module_name=args.module_name, flatten_acts=False, output_type="ndarray")
    feature_dim = raw_features.shape[2] # Dimensione corretta è la terza
else:
    raw_features = extractor.extract_features(batches=[temp_batch_images], module_name=args.module_name, flatten_acts=True, output_type="ndarray")
    feature_dim = raw_features.shape[1]
# --- END FIX ---
"""

print(f"Detected final dataset shape: ({total_rows}, {feature_dim})")

# 2. Creiamo il file HDF5 finale e i dataset al suo interno
final_hdf5_path = os.path.join(run_path, FINAL_HDF5_FILENAME)
with h5py.File(final_hdf5_path, 'w') as h5f:
    # Creiamo il dataset per le feature. `chunks=True` è per l'efficienza.
    dset_features = h5f.create_dataset('features', 
                                       shape=(total_rows, feature_dim), 
                                       dtype='f4', # float32
                                       chunks=True)
    
    # Creiamo il dataset per i path di stringhe a lunghezza variabile
    dt = h5py.special_dtype(vlen=str)
    dset_paths = h5f.create_dataset('image_paths', 
                                    shape=(total_rows,), 
                                    dtype=dt,
                                    chunks=True)
    
    print(f"HDF5 file created at: {final_hdf5_path}")
    print("Starting data processing loop...")

    # 3. Loop di estrazione e scrittura diretta nel file HDF5
    batch_count = 0
    progress_bar = tqdm(dataloader, desc="Extracting Features", total=len(dataloader))
    #for batch_idx, batch in enumerate(dataloader):

    for batch_idx, (batch_images, batch_paths) in enumerate(progress_bar):
        if args.test_mode and batch_idx >= args.test_batches:
            break

    try:
        # Sposta solo il tensore delle immagini sul dispositivo corretto
        batch_images = batch_images.to(device)

        # Estrazione delle feature usando il batch di immagini
        if args.model_type == "andrea_dino":
            raw_batch_features = extractor.extract_features(batches=[batch_images], module_name=args.module_name, flatten_acts=False, output_type="ndarray")
            batch_features = raw_batch_features[:, 0, :]
        else:
            # Questa è la logica per il tuo modello "cecconello_ssl"
            batch_features = extractor.extract_features(batches=[batch_images], module_name=args.module_name, flatten_acts=True, output_type="ndarray")
        
        # Calcolo degli indici per la scrittura
        start_index = batch_idx * args.batch_size
        end_index = start_index + len(batch_features)

        # SCRITTURA DIRETTA NEL FILE HDF5
        dset_features[start_index:end_index] = batch_features
        
        # Ora 'batch_paths' è una tupla/lista con la dimensione corretta (es. 1024)
        # ricevuta direttamente dal DataLoader, e l'errore di broadcast è risolto.
        dset_paths[start_index:end_index] = batch_paths

    except Exception as e:
        print(f"Error processing batch {batch_idx}: {e}")
        import traceback
        traceback.print_exc()

    batch_count += 1
        #torch.cuda.empty_cache()
    """OLD CODE:
    for batch_idx, batch in enumerate(progress_bar):
        if args.test_mode and batch_idx >= args.test_batches: break
        #if batch_idx >= 10: break
        print(f"Processing batch {batch_idx+1}/{len(dataloader)}")
        
        try:
            # Estrazione (logica invariata)
            if args.model_type == "andrea_dino":
                raw_batch_features = extractor.extract_features(batches=[batch], module_name=args.module_name, flatten_acts=False, output_type="ndarray")
                batch_features = raw_batch_features[:, 0, :]
            else:
                batch_features = extractor.extract_features(batches=[batch], module_name=args.module_name, flatten_acts=True, output_type="ndarray")
            
            # Recupero path
            batch_paths = [dataset.info.iloc[batch_idx * args.batch_size + i]["file_path"] for i in range(len(batch[0]))]
            
            # Calcolo indici per la scrittura
            start_index = batch_idx * args.batch_size
            end_index = start_index + len(batch_features)

            # SCRITTURA DIRETTA NEL FILE HDF5 (RAM minima usata)
            dset_features[start_index:end_index] = batch_features
            dset_paths[start_index:end_index] = batch_paths

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}"); traceback.print_exc()

        batch_count += 1
        #torch.cuda.empty_cache()
    """
# Il blocco `with h5py.File(...)` ha chiuso il file in modo sicuro.
print("\n--- Feature extraction complete. HDF5 file is finalized and ready. ---")


# ──────────────────────────────────────────────────────────────────────────────
#   FINAL METADATA & LOGGING
# ──────────────────────────────────────────────────────────────────────────────
metadata["feature_count"] = total_rows
metadata["feature_dimension"] = feature_dim
metadata["hdf5_data_path"] = final_hdf5_path # Aggiorniamo il metadato
metadata["processed_batches"] = batch_count

with open(metadata_file_path, 'w') as f: json.dump(metadata, f, indent=2)

print("\n==== PROCESSING SUMMARY ====")
print(f"Mode: {'TEST' if args.test_mode else 'PRODUCTION'}")
print(f"Batches processed: {batch_count}")
print(f"Total samples: {metadata['feature_count']}")
print(f"Feature dimension: {metadata['feature_dimension']}")
print(f"Final Data file (HDF5): {final_hdf5_path}")
print(f"Metadata file: {metadata_file_path}")
print("=============================\n")

log_on_mlflow(MLFLOW_EXPERIMENT_NAME, MLFLOW_RUN_NAME, metadata, metadata_file_path)