from datetime import datetime
from utils import format_readable_run_name, compute_config_hash, log_on_mlflow
from parsers.extract_features_parser import get_parser

import os
import mlflow
import json
import numpy as np
import torchvision.transforms as T
import torch
import re

from collections import OrderedDict
from thingsvision import get_extractor_from_model, get_extractor
from torchvision.models import resnet18, resnet50
from thingsvision.utils.data import DataLoader
from thingsvision.utils.storing import save_features
from data.custom import CustomUnlabeledDatasetWithPath

MLFLOW_EXPERIMENT_NAME = "extract_features"
MLFLOW_RUN_NAME = "extract_features"
ROOT_FOLDER = "outputs/features"
OUTPUT_FOLDER_BASENAME = "features"
METADATA_FILE_BASENAME = "metadata.json"
FEATURES_LIST_FILENAME = "features.npy"

def parse_model_parameters(param_list):
    """Parse model parameters from a list of key=value strings."""
    return {kv.split('=')[0]: kv.split('=')[1] for kv in param_list if '=' in kv}

# Parse arguments
args = get_parser()
metadata = vars(args)
print(metadata)
# Create unique hash for config and setup folders
config_hash = compute_config_hash(metadata, exclude_keys=["cuda_devices"])
metadata["timestamp"] = datetime.now().isoformat()
metadata["config_hash"] = config_hash
metadata["root_folder"] = ROOT_FOLDER
metadata["features_filename"] = FEATURES_LIST_FILENAME

# Create readable folder name
readable_name_params = {}
readable_name_params["model"] = metadata.get("model_name", "")
readable_name_params["variant"] = metadata.get("variant", "")
metadata["run_folder"] = format_readable_run_name(OUTPUT_FOLDER_BASENAME, readable_name_params, config_hash)

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

# Get source data from MLflow
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
"""
# Load cutout information
with open(merged_cutout_path, "r") as f:
    merged_cutouts_info = json.load(f)
    n_cutouts = len(merged_cutouts_info)
    print(f"Found {n_cutouts} cutouts to process")

# MAIN LOGIC 
# Set up device
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
elif args.normalization == "minmax":
    norm = MinMaxNormalize()

# Set up transformations
transforms = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x: x.repeat(3,1,1)),
    norm,
    T.Resize(args.resize)
])

# Create dataset
print(f"Loading dataset from {merged_cutout_folder_path}")
dataset = CustomUnlabeledDatasetWithPath(
    data_path=merged_cutout_folder_path,
    loader_type="npy",
    transforms=transforms,
    datalist=merged_cutouts_info_filename
)
print(f"Dataset size: {len(dataset)}")

# Update metadata with test mode info
if args.test_mode:
    metadata["test_mode"] = True
    metadata["test_batches"] = args.test_batches
    print(f"TEST MODE ACTIVE: Limited to {args.test_batches} batches")
else:
    metadata["test_mode"] = False
    print("FULL MODE: Processing all batches")

# Create DataLoader
dataloader = DataLoader(
    dataset=dataset,
    batch_size=args.batch_size,
    backend=extractor.get_backend()
)

print("Starting feature extraction...")
all_features = []
all_image_paths = []
batch_count = 0
batch_times = []  # Lista per memorizzare i tempi di elaborazione dei batch (solo in test mode)

# Extract features in a single pass
for batch_idx, batch in enumerate(dataloader):
    # Check if we're in test mode and should stop
    if args.test_mode and batch_idx >= args.test_batches:
        break
    
    print(f"Processing batch {batch_idx+1}")
    
    start_time = datetime.now()  # Inizio del tempo di elaborazione del batch
    
    try:
        # Extract features for this batch
        batch_features = extractor.extract_features(
            batches=[batch],
            module_name=args.module_name,
            flatten_acts=True,
            output_type="ndarray",
        )
        
        # Get paths for this batch
        batch_paths = []
        for i in range(len(batch)):
            try:
                # Get the absolute index in the dataset based on the batch index
                # Note: this assumes sequential sampling (not shuffled)
                absolute_idx = batch_idx * args.batch_size + i
                
                # Access the file_path directly from the dataset's info
                if absolute_idx < len(dataset):
                    file_path = dataset.info.iloc[absolute_idx]["file_path"]
                    batch_paths.append(file_path)
                else:
                    # Handle edge case where the last batch might be smaller
                    batch_paths.append(f"out_of_range_idx_{absolute_idx}")
                    
            except Exception as e:
                # Fallback if path extraction fails
                batch_paths.append(f"unknown_batch_{batch_idx}_item_{i}")
                print(f"Could not retrieve path for batch {batch_idx}, item {i}: {e}")
        
        # Store features and paths
        all_features.append(batch_features)
        all_image_paths.extend(batch_paths)
        
        # Print some info about the first batch for debugging
        if batch_idx == 0:
            print(f"First batch shape: {batch_features.shape}")
            print(f"Feature dimension: {batch_features.shape[1]}")
        
        # Clean up to free memory
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error processing batch {batch_idx}: {e}")
        import traceback
        traceback.print_exc()
    
    end_time = datetime.now()  # Fine del tempo di elaborazione del batch
    batch_time = (end_time - start_time).total_seconds()
    
    if args.test_mode:
        batch_times.append(batch_time)
    
    batch_count += 1

# Calcola il tempo medio per batch e stima il tempo totale (solo in test mode)
if args.test_mode:
    if batch_times:
        avg_batch_time = sum(batch_times) / len(batch_times)
        total_batches = len(dataloader)
        estimated_total_time = avg_batch_time * total_batches
        print(f"Tempo medio per batch: {avg_batch_time:.2f} secondi")
        print(f"Numero totale di batch: {total_batches}")
        print(f"Tempo totale stimato: {estimated_total_time:.2f} secondi")
    else:
        print("Nessun batch elaborato, impossibile stimare il tempo.")

# Combine all extracted features
print("Combining all features...")
features = np.vstack(all_features)
print(f"Combined features shape: {features.shape}")
"""
# Save features to disk
features_path = os.path.join(run_path, FEATURES_LIST_FILENAME)
#np.save(features_path, features)
#print(f"Features saved to {features_path}")

"""
# Save image paths
paths_file = os.path.join(run_path, "image_paths.json")
with open(paths_file, 'w') as f:
    json.dump(all_image_paths, f)
print(f"Image paths saved to {paths_file}")
"""

# Update metadata
#metadata["feature_count"] = features.shape[0]
#metadata["feature_dimension"] = features.shape[1]
metadata["features_path"] = features_path
#metadata["image_paths_file"] = paths_file
#metadata["processed_batches"] = batch_count

# Save metadata
#with open(metadata_file_path, 'w') as f:
#    json.dump(metadata, f, indent=2)

# Final summary
print("\n==== PROCESSING SUMMARY ====")
print(f"Mode: {'TEST' if args.test_mode else 'PRODUCTION'}")
#print(f"Batches processed: {batch_count}")
#print(f"Total samples: {features.shape[0]}")
#print(f"Feature dimension: {features.shape[1]}")
#print(f"Features file: {features_path}")
#print(f"Paths file: {paths_file}")
#print(f"Metadata file: {metadata_file_path}")
print("=============================\n")

log_on_mlflow(MLFLOW_EXPERIMENT_NAME, MLFLOW_RUN_NAME, metadata, metadata_file_path)