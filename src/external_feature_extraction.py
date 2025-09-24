# ──────────────────────────────────────────────────────────────────────────────
#   IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
from datetime import datetime
# Rimosso log_on_mlflow da qui
from utils import format_readable_run_name, compute_config_hash
from parsers.extract_features_external_parser import get_parser

import sys
from tqdm import tqdm 
import os
# import mlflow # RIMOSSO
import json
import numpy as np
import torchvision.transforms as T
import torch
import re
import traceback
import h5py

from transformers import AutoConfig, AutoModel
from collections import OrderedDict
from thingsvision import get_extractor_from_model, get_extractor
from torchvision.models import resnet18, resnet50
from data.custom import CustomUnlabeledDatasetWithPath
from torch.utils.data import DataLoader as PyTorchDataLoader

# La classe MinMaxNormalize rimane invariata
class MinMaxNormalize(object):
    def __call__(self, tensor):
        min_val, max_val = tensor.min(), tensor.max()
        if max_val - min_val > 0:
            tensor = (tensor - min_val) / (max_val - min_val)
        else:
            tensor.fill_(0) 
        return tensor
    def __repr__(self): return self.__class__.__name__ + '()'

def parse_model_parameters(param_list):
    return {kv.split('=')[0]: kv.split('=')[1] for kv in param_list if '=' in kv}

# ──────────────────────────────────────────────────────────────────────────────
#   CONSTANTS & SETUP
# ──────────────────────────────────────────────────────────────────────────────
# Costanti MLflow RIMOSSE
ROOT_FOLDER = "outputs/features"
OUTPUT_FOLDER_BASENAME = "features"
METADATA_FILE_BASENAME = "metadata.json"
FINAL_HDF5_FILENAME = "features_data_fast.h5"

# ──────────────────────────────────────────────────────────────────────────────
#   ARGUMENT PARSING & FOLDER SETUP
# ──────────────────────────────────────────────────────────────────────────────
args = get_parser()
metadata = vars(args)

config_hash = compute_config_hash(metadata, exclude_keys=["cuda_devices", "input_folder", "info_json_name"])
metadata["timestamp"] = datetime.now().isoformat()
metadata["config_hash"] = config_hash
metadata["root_folder"] = ROOT_FOLDER

readable_name_params = {"model": metadata.get("model_name", ""), "variant": metadata.get("variant", "")}
metadata["run_folder"] = format_readable_run_name(OUTPUT_FOLDER_BASENAME, readable_name_params, config_hash)
metadata["features_filename"] = FINAL_HDF5_FILENAME

run_path = os.path.join(metadata["root_folder"], metadata["run_folder"])
metadata_file_path = os.path.join(run_path, METADATA_FILE_BASENAME)

print(f"Output directory: {run_path}")
if not os.path.exists(run_path):
    os.makedirs(run_path)
else:
    if os.listdir(run_path):
        print(f"Experiment already exists. To recompute, delete folder {run_path}")
        sys.exit()
    else:
        print(f"Folder {run_path} exists but is empty, proceeding.")

# ──────────────────────────────────────────────────────────────────────────────
#   DATA INPUT SETUP (SEMPLIFICATO)
# ──────────────────────────────────────────────────────────────────────────────
print(f"Reading data from local folder.")
if not os.path.isdir(args.input_folder):
    print(f"Error: Input folder '{args.input_folder}' does not exist.")
    sys.exit(1)
    
merged_cutout_folder_path = args.input_folder
merged_cutouts_info_filename = args.info_json_name
merged_cutout_path = os.path.join(merged_cutout_folder_path, merged_cutouts_info_filename)

if not os.path.isfile(merged_cutout_path):
    print(f"Error: Info file '{merged_cutout_path}' not found in the input folder.")
    sys.exit(1)

with open(merged_cutout_path, "r") as f:
    merged_cutouts_info = json.load(f)
    n_cutouts = len(merged_cutouts_info)
    print(f"Found {n_cutouts} cutouts to process from '{merged_cutout_folder_path}'")

# ──────────────────────────────────────────────────────────────────────────────
#   MODEL, EXTRACTOR, AND TRANSFORM INITIALIZATION (INVARIATO)
# ──────────────────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# (Il resto di questa sezione è identico e corretto, lo ometto per brevità)
model_parameters = parse_model_parameters(args.model_parameters)
if args.model_type == "cecconello_ssl":
    if not args.ckpt_path or not args.model_name:
        print("Errore: Per model_type 'cecconello_ssl', sono richiesti sia --model_name che --ckpt_path.")
        exit(1)
    # ... resto del codice per il modello ...
    if args.model_name == "resnet18":
        model = resnet18(weights=None)
    elif args.model_name == "resnet50":
        model = resnet50(weights=None)
    else:
        print(f"Errore: Modello '{args.model_name}' non supportato. Usare 'resnet18' o 'resnet50'.")
        exit(1)
    model.fc = torch.nn.Identity()
    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    backbone_state_dict = OrderedDict([(k.replace("backbone.", ""), v) for k, v in checkpoint["state_dict"].items() if k.startswith("backbone.")])
    model.load_state_dict(backbone_state_dict)
    extractor = get_extractor_from_model(model=model, device=device, backend='pt')
# ... le altre condizioni per 'andrea_dino' e 'thingsvision' rimangono uguali ...
elif args.model_type == "andrea_dino":
    USE_GRAYSCALE = True
    config = AutoConfig.from_pretrained(args.model_name, num_channels=1 if USE_GRAYSCALE else 3)
    model = AutoModel.from_pretrained(args.model_name, config=config, ignore_mismatched_sizes=USE_GRAYSCALE).to(device)
    extractor = get_extractor_from_model(model=model, device=device, backend='pt')
elif args.model_type == "thingsvision":
    extractor = get_extractor(model_name=args.model_name, source=args.source, device=device, pretrained=True, model_parameters=model_parameters)


# Pipeline di trasformazioni (INVARIATA)
transform_list = [T.ToTensor()]
if args.normalization == "minmax":
    transform_list.append(MinMaxNormalize())
else:
    # ... logica di normalizzazione ...
    if args.normalization == "imagenet": mean, std = [0.45], [0.225]
    elif args.normalization == "mean05std05": mean, std = [0.5], [0.5]
    elif args.normalization == "mean05std0225": mean, std = [0.5], [0.225]
    elif args.normalization == "meanstdLoTSS": mean, std = [2e-05], [0.003]
    else:
        print(f"!! WARNING: Normalizzazione '{args.normalization}' non riconosciuta, uso 'minmax'.")
        mean, std = None, None
        transform_list.append(MinMaxNormalize())
    if mean is not None:
        transform_list.append(T.Normalize(mean, std))
if args.model_input_channels == 3: transform_list.append(T.Lambda(lambda x: x.repeat(3, 1, 1)))
#transform_list.append(T.Resize(args.resize))
transform_list.append(T.Resize((args.resize, args.resize)))
transforms = T.Compose(transform_list)
print("\n--- Final Transform Pipeline ---\n", transforms, "\n" + "-"*30)

# ──────────────────────────────────────────────────────────────────────────────
#   DATASET & DATALOADER SETUP (INVARIATO)
# ──────────────────────────────────────────────────────────────────────────────
dataset = CustomUnlabeledDatasetWithPath(
    data_path=merged_cutout_folder_path,
    loader_type="npy",
    transforms=transforms,
    datalist=merged_cutouts_info_filename
)
NUM_WORKERS = min(os.cpu_count(), 8) if device == 'cuda' else 0
dataloader = PyTorchDataLoader(
    dataset=dataset, batch_size=args.batch_size, num_workers=NUM_WORKERS,
    pin_memory=True, prefetch_factor=2 if NUM_WORKERS > 0 else None,
    shuffle=False, drop_last=False
)

# ──────────────────────────────────────────────────────────────────────────────
#   FEATURE EXTRACTION WITH DIRECT HDF5 WRITING (INVARIATO)
# ──────────────────────────────────────────────────────────────────────────────
print("\n--- Starting feature extraction ---")
total_rows = len(dataset)
temp_batch_images, _ = next(iter(dataloader)) 
if args.model_type == "andrea_dino":
    raw_features = extractor.extract_features(batches=[temp_batch_images], module_name=args.module_name, flatten_acts=False, output_type="ndarray")
    feature_dim = raw_features.shape[2] 
else:
    raw_features = extractor.extract_features(batches=[temp_batch_images], module_name=args.module_name, flatten_acts=True, output_type="ndarray")
    feature_dim = raw_features.shape[1]
print(f"Detected final dataset shape: ({total_rows}, {feature_dim})")

final_hdf5_path = os.path.join(run_path, FINAL_HDF5_FILENAME)
with h5py.File(final_hdf5_path, 'w') as h5f:
    dset_features = h5f.create_dataset('features', shape=(total_rows, feature_dim), dtype='f4', chunks=True)
    dt = h5py.special_dtype(vlen=str)
    dset_paths = h5f.create_dataset('image_paths', shape=(total_rows,), dtype=dt, chunks=True)
    
    batch_count = 0
    progress_bar = tqdm(dataloader, desc="Extracting Features", total=len(dataloader))
    for batch_idx, (batch_images, batch_paths) in enumerate(progress_bar):
        if args.test_mode and batch_idx >= args.test_batches: break
        try:
            batch_images = batch_images.to(device)
            if args.model_type == "andrea_dino":
                raw_batch_features = extractor.extract_features(batches=[batch_images], module_name=args.module_name, flatten_acts=False, output_type="ndarray")
                batch_features = raw_batch_features[:, 0, :]
            else:
                batch_features = extractor.extract_features(batches=[batch_images], module_name=args.module_name, flatten_acts=True, output_type="ndarray")
            
            start_index = batch_idx * args.batch_size
            end_index = start_index + len(batch_features)
            dset_features[start_index:end_index] = batch_features
            dset_paths[start_index:end_index] = batch_paths
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            traceback.print_exc()
        batch_count += 1

print("\n--- Feature extraction complete. HDF5 file is ready. ---")

# ──────────────────────────────────────────────────────────────────────────────
#   FINAL METADATA (SEMPLIFICATO)
# ──────────────────────────────────────────────────────────────────────────────
metadata["feature_count"] = total_rows
metadata["feature_dimension"] = feature_dim
metadata["hdf5_data_path"] = final_hdf5_path
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

# Chiamata a log_on_mlflow RIMOSSA