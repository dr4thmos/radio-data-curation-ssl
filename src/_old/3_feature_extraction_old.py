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
    """Parsa i parametri del modello da una lista di stringhe key=value."""
    return {kv.split('=')[0]: kv.split('=')[1] for kv in param_list if '=' in kv}

args = get_parser()
metadata=vars(args)

config_hash = compute_config_hash(metadata, exclude_keys=["cuda_devices"])
metadata["timestamp"] = datetime.now().isoformat()
metadata["config_hash"] = config_hash
metadata["root_folder"] = ROOT_FOLDER
metadata["features_filename"] = FEATURES_LIST_FILENAME

readable_name_params = {}
readable_name_params["model"] = metadata.get("model_name", "")
readable_name_params["variant"] = metadata.get("variant", "")
metadata["run_folder"] = format_readable_run_name(OUTPUT_FOLDER_BASENAME, readable_name_params, config_hash)

run_path = os.path.join(metadata["root_folder"], metadata["run_folder"])
metadata_file_path = os.path.join(run_path, METADATA_FILE_BASENAME)

print(run_path)
if not os.path.exists(run_path):
    os.makedirs(run_path)
else:
    if os.listdir(run_path):  # Controlla se la cartella non è vuota
        print(f"Esperimento già presente e non vuoto, per ricomputarlo eliminare la cartella {run_path}")
        exit()
    else:
        print(f"La cartella {run_path} esiste ma è vuota, procedo con l'esperimento.")

"""
features = extractor.extract_features(
    batches=batches,
    module_name=args.module_name,
    flatten_acts=True,
    output_type="ndarray", # or "tensor" (only applicable to PyTorch models of which CLIP and DINO are ones!)
)

 
save_features(features, out_path=out_path, file_format=metadata["output_file_format"]) # file_format can be set to "npy", "txt", "mat", "pt", or "hdf5"
"""

try:
    run = mlflow.get_run(metadata["source_id"]) # 42648460a62c40b1b8a0fd69e01d9510
    merged_cutouts_root_folder = run.data.params.get("root_folder")
    merged_cutouts_run_folder = run.data.params.get("run_folder")
    merged_cutouts_info_filename = run.data.params.get("merged_cutouts_info_filename")
    merged_cutout_folder_path = os.path.join(merged_cutouts_root_folder, merged_cutouts_run_folder)
    merged_cutout_path = os.path.join(merged_cutouts_root_folder, merged_cutouts_run_folder, merged_cutouts_info_filename)

except mlflow.exceptions.MlflowException as e:
    print(f"Errore nel recuperare il cutout_id {metadata['source_id']}: {e}")

with open(merged_cutout_path, "r") as f:
    merged_cutouts_info = json.load(f) #rimuovere?
    n_cutouts = len(merged_cutouts_info)

""" MAIN LOGIC """
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_parameters = parse_model_parameters(args.model_parameters)
if args.model_type == "cecconello_ssl":
    model_metadata_path = os.path.join(os.path.dirname(args.ckpt_path), "args.json")

    with open(model_metadata_path, "r") as f:
        model_metadata = json.load(f)

    out_path = os.path.join(args.features_folder, model_metadata["name"])
    
    #ckpt_path = "/home/tcecconello/radioimgs/radio-data-curation-ssl/model_weights/uja9qvb7/byol_hulk_aug_minmax_model_resnet18-uja9qvb7-ep=100.ckpt"
    #"/home/tcecconello/radioimgs/radio-data-curation-ssl/model_weights/uja9qvb7.ckpt"
    if args.model_name == "resnet18":
        model = resnet18(weights=None)
    if args.model_name == "resnet50":
        model = resnet50(weights=None)
    #print(model)
    model.fc = torch.nn.Identity()
    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False) # checkpoint = torch.load(ckpt_path, map_location={'cuda:0': device})


    #print(checkpoint["state_dict"].keys())
    backbone_state_dict = OrderedDict(
        [(k.replace("backbone.", ""), v) for k, v in checkpoint["state_dict"].items() if k.startswith("backbone.")]
    )

    model.load_state_dict(backbone_state_dict)

    # you can also pass a custom preprocessing function that is applied to every 
    # image before extraction
    #transforms = model_weights.transforms()

    # provide the backend of the model (either 'pt' or 'tf')
    backend = 'pt'

    extractor = get_extractor_from_model(
    model=model, 
    device=device,
    #transforms=transforms,
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

    #extractor.show_model()

norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
if args.normalization == "imagenet":
    norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
if args.normalization == "mean05std05":
    norm = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
if args.normalization == "mean05std025":
    norm = T.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
if args.normalization == "meanstdLoTSS":
    norm = T.Normalize((2e-05, 2e-05, 2e-05), (0.003, 0.003, 0.003))
if args.normalization == "minmax":
    norm = MinMaxNormalize()

transforms = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x: x.repeat(3,1,1)),
    norm,
    T.Resize(args.resize)
])

print(merged_cutout_folder_path)
print(merged_cutouts_info_filename)
dataset = CustomUnlabeledDatasetWithPath(
    data_path=merged_cutout_folder_path,
    loader_type="npy",
    transforms=transforms,
    datalist=merged_cutouts_info_filename
    #extractor.get_transformations(resize_dim=256, crop_dim=224),
)
print(dataset)
batches = DataLoader(
    dataset=dataset,
    batch_size=args.batch_size,
    #num_workers=0,
    backend=extractor.get_backend() # backend framework of model
)
"""
features = extractor.extract_features(
    batches=batches,
    module_name=args.module_name,
    flatten_acts=True,
    output_type="ndarray", # or "tensor" (only applicable to PyTorch models of which CLIP and DINO are ones!)
)

save_features(features, out_path=out_path, file_format='npy') # file_format can be set to "npy", "txt", "mat", "pt", or "hdf5"
"""
# Modifiche per migliorare la gestione della memoria

# 1. Modifica la funzione extract_features per salvare incrementalmente
num_batches_to_test = 3
features_list = []
for i, batch in enumerate(batches):
    if i >= num_batches_to_test:
        break
    # Estrai feature per questo batch
    batch_features = extractor.extract_features(
        batches=[batch],  # Passa un singolo batch alla volta
        module_name=args.module_name,
        flatten_acts=True,
        output_type="ndarray",
    )
    
    # Salva le feature del batch su disco
    batch_path = os.path.join(run_path, f"batch_{i}_features.npy")
    np.save(batch_path, batch_features)
    
    # Libera memoria
    del batch_features
    torch.cuda.empty_cache()  # Pulisci la memoria CUDA
    
    print(f"Batch {i} completato e salvato")


# Determina la dimensione del vettore di feature dal primo batch
first_batch = np.load(os.path.join(run_path, "batch_0_features.npy"))
feature_dim = first_batch.shape[1]  # Dimensione del vettore di feature

# Conta il numero totale di campioni
total_samples = 0
batch_files = sorted([f for f in os.listdir(run_path) if f.startswith("batch_") and f.endswith("_features.npy")])
for batch_file in batch_files:
    batch_path = os.path.join(run_path, batch_file)
    batch_data = np.load(batch_path)
    total_samples += batch_data.shape[0]

# Crea un memmap file per salvare tutte le feature
features_path = os.path.join(run_path, FEATURES_LIST_FILENAME)
all_features = np.memmap(features_path, dtype='float32', mode='w+', shape=(total_samples, feature_dim))

# Copia incrementalmente tutti i batch nel file memmap
start_idx = 0
for batch_file in batch_files:
    batch_path = os.path.join(run_path, batch_file)
    batch_data = np.load(batch_path)
    batch_size = batch_data.shape[0]
    
    # Copia i dati nella posizione corretta
    all_features[start_idx:start_idx+batch_size] = batch_data
    start_idx += batch_size
    
    # Rimuovi il file batch dopo averlo copiato
    os.remove(batch_path)

# Flush dei dati sul disco
all_features.flush()

log_on_mlflow(MLFLOW_EXPERIMENT_NAME, MLFLOW_RUN_NAME, metadata, metadata_file_path)