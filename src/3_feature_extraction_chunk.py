# ──────────────────────────────────────────────────────────────────────────────
#   IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
from datetime import datetime
from utils import format_readable_run_name, compute_config_hash, log_on_mlflow
from parsers.extract_features_parser import get_parser

import sys
import os
import mlflow
import json
import numpy as np
import torchvision.transforms as T
import torch
import re
import traceback
import h5py # <-- LO STRUMENTO GIUSTO

from transformers import (
    AutoConfig, AutoModel, AutoModelForImageClassification, AutoImageProcessor
)
from collections import OrderedDict
from thingsvision import get_extractor_from_model, get_extractor
from torchvision.models import resnet18, resnet50
from thingsvision.utils.data import DataLoader
from data.custom import CustomUnlabeledDatasetWithPath

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
FINAL_HDF5_FILENAME = "features_data.h5"

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
    model_metadata_path = os.path.join(os.path.dirname(args.ckpt_path), "args.json")
    
    with open(model_metadata_path, "r") as f:
        model_metadata = json.load(f)
    
    out_path = os.path.join(args.features_folder, model_metadata["name"])
    
    if args.model_name == "resnet18":
        model = resnet18(weights=None)
    if args.model_name == "resnet50":
        model = resnet50(weights=None)
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

if USE_GRAYSCALE:
        # Set up normalization based on arguments
    norm = T.Normalize((0.45), (0.225))  # Default ImageNet normalization
    print(f"Using normalization: {args.normalization}")
    if args.normalization == "imagenet":
        print(f"Using normalization: {args.normalization}")
        norm = T.Normalize((0.45), (0.225))
    elif args.normalization == "mean05std05":
        print(f"Using normalization: {args.normalization}")
        norm = T.Normalize((0.5), (0.5))
    elif args.normalization == "mean05std0225":
        print(f"Using normalization: {args.normalization}")
        norm = T.Normalize((0.5), (0.225))
    elif args.normalization == "meanstdLoTSS":
        print(f"Using normalization: {args.normalization}")
        norm = T.Normalize((2e-05), (0.003))
    # Aggiungo un controllo per MinMaxNormalize nel caso non sia definito
    # elif args.normalization == "minmax":
    #     norm = MinMaxNormalize()

    # Set up transformations
    transforms = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(1,1,1)),
        norm,
        T.Resize(args.resize)
    ])
else:
    # Set up normalization based on arguments
    norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Default ImageNet normalization
    print(f"Using normalization: {args.normalization}")
    if args.normalization == "imagenet":
        print(f"Using normalization: {args.normalization}")
        norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    elif args.normalization == "mean05std05":
        print(f"Using normalization: {args.normalization}")
        norm = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    elif args.normalization == "mean05std0225":
        print(f"Using normalization: {args.normalization}")
        norm = T.Normalize((0.5, 0.5, 0.5), (0.225, 0.225, 0.225))
    elif args.normalization == "meanstdLoTSS":
        print(f"Using normalization: {args.normalization}")
        norm = T.Normalize((2e-05, 2e-05, 2e-05), (0.003, 0.003, 0.003))
    # Aggiungo un controllo per MinMaxNormalize nel caso non sia definito
    # elif args.normalization == "minmax":
    #     norm = MinMaxNormalize()

    # Set up transformations
    transforms = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3,1,1)),
        norm,
        T.Resize(args.resize)
    ])


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

dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, backend=extractor.get_backend())

# ──────────────────────────────────────────────────────────────────────────────
#   FEATURE EXTRACTION WITH DIRECT HDF5 WRITING
# ──────────────────────────────────────────────────────────────────────────────
print("\n--- Starting feature extraction with robust HDF5 writing ---")

# 1. Calcoliamo in anticipo le dimensioni finali
total_rows = len(dataset)
# Eseguiamo un batch di prova per ottenere la dimensione delle feature
temp_batch = next(iter(dataloader))
if args.model_type == "andrea_dino":
    raw_features = extractor.extract_features(batches=[temp_batch], module_name=args.module_name, flatten_acts=False, output_type="ndarray")
    feature_dim = raw_features.shape[2] # Dimensione corretta è la terza
else:
    raw_features = extractor.extract_features(batches=[temp_batch], module_name=args.module_name, flatten_acts=True, output_type="ndarray")
    feature_dim = raw_features.shape[1]
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
    for batch_idx, batch in enumerate(dataloader):
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
        torch.cuda.empty_cache()

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