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

def print_file_size(filepath):
    if os.path.exists(filepath):
        size_bytes = os.path.getsize(filepath)
        size_mb = size_bytes / (1024 * 1024)
        print(f"Current file size of {filepath}: {size_mb:.2f} MB")
    else:
        print(f"File {filepath} does not exist yet")

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
)
print(dataset)

# Funzione per creare un nuovo DataLoader con le stesse impostazioni
def create_dataloader():
    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        backend=extractor.get_backend() # backend framework of model
    )

# Crea il primo DataLoader per la prima passata
batches = create_dataloader()

# Estrazione e salvataggio incrementale delle features con tracciamento dei path
features_path = os.path.join(run_path, FEATURES_LIST_FILENAME)
paths_file = os.path.join(run_path, "image_paths.json")
batch_count = 0
total_samples = 0
feature_dim = None
all_image_paths = []

print(args.test_mode)

# Modalità test: aggiungiamo l'informazione nei metadati
if args.test_mode:
    metadata["test_mode"] = True
    metadata["test_batches"] = args.test_batches
    print(f"MODALITÀ TEST ATTIVA: Elaborazione limitata a {args.test_batches} batch")
else:
    metadata["test_mode"] = False
    print("MODALITÀ COMPLETA: Elaborazione di tutti i batch")

# Prima passata: contiamo i campioni totali e determiniamo la dimensione delle features
print("Fase 1: Conteggio campioni e determinazione dimensione features")
for batch_idx, batch in enumerate(batches):
    # In modalità test, limita il conteggio
    if args.test_mode and batch_idx >= args.test_batches:
        break
        
    # Estrai feature solo per il primo batch per determinare la dimensione
    if batch_idx == 0:
        batch_features = extractor.extract_features(
            batches=[batch],
            module_name=args.module_name,
            flatten_acts=True,
            output_type="ndarray",
        )
        print("Dimensione features estratte:", batch_features.shape)
        feature_dim = batch_features.shape[1]
        total_samples += batch_features.shape[0]
        # Salva temporaneamente per testare
        temp_batch_path = os.path.join(run_path, f"temp_batch_features.npy")
        np.save(temp_batch_path, batch_features)
        del batch_features
        
        # Raccogliamo i percorsi delle immagini dal primo batch
        for i in range(len(batch)):
            try:
                print(i)
                # Recupera il percorso del file dall'oggetto batch
                file_path = batch.dataset.info.iloc[batch.indices[i]]["file_path"]
                all_image_paths.append(file_path)
            except (AttributeError, IndexError, KeyError) as e:
                # Fallback nel caso in cui la struttura del dataset sia diversa
                all_image_paths.append(f"unknown_batch_{batch_idx}_item_{i}")
                print(f"Impossibile recuperare il percorso per batch {batch_idx}, item {i}: {e}")
    else:
        # Per gli altri batch, calcola correttamente il numero di campioni
        if isinstance(batch, list):
            batch_size = len(batch[0])
        else:
            # Get the batch size directly from the batch
            batch_size = len(batch)
        total_samples += batch_size
        print(f"Batch {batch_idx}: aggiunto {batch_size} campioni, totale: {total_samples}")
    
    batch_count += 1
    torch.cuda.empty_cache()  # Pulisci la memoria CUDA

# Assicuriamoci che batch_count rifletta il numero reale di batch che processeremo
if args.test_mode:
    batch_count = min(batch_count, args.test_batches)

print(f"Totale campioni da elaborare: {total_samples}, Dimensione features: {feature_dim}")
print(f"Totale batch da elaborare: {batch_count}")

# Crea il file memmap con la dimensione corretta
print(f"Fase 2: Creazione file memmap per {total_samples} campioni")
all_features = np.memmap(features_path, dtype='float32', mode='w+', 
                         shape=(total_samples, feature_dim))
print_file_size(features_path)

# Carica e salva il primo batch già estratto
first_batch = np.load(os.path.join(run_path, "temp_batch_features.npy"))
first_batch_size = first_batch.shape[0]
all_features[:first_batch_size] = first_batch
print_file_size(features_path)
os.remove(os.path.join(run_path, "temp_batch_features.npy"))
start_idx = first_batch_size

# Crea un nuovo DataLoader per la seconda passata
print(f"Fase 3: Estrazione e salvataggio incrementale delle features per {batch_count} batch")
# Ricrea l'iteratore DataLoader per la seconda passata
# Before creating the second DataLoader
del batches  # Explicitly delete the old DataLoader
torch.cuda.empty_cache()  # Clear CUDA cache
print("Recreating DataLoader for feature extraction...")
batches = create_dataloader()

# Seconda passata: estrazione e salvataggio incrementale
for batch_idx, batch in enumerate(batches):
    # In modalità test, limita l'elaborazione
    if args.test_mode and batch_idx >= args.test_batches:
        break
        
    # Salta il primo batch che abbiamo già elaborato
    if batch_idx == 0:
        continue
        
    print(f"Elaborazione batch {batch_idx+1}/{batch_count} " + 
          f"({'test' if args.test_mode else 'produzione'})")
    try:
        # Estrai feature per questo batch
        batch_features = extractor.extract_features(
            batches=[batch],
            module_name=args.module_name,
            flatten_acts=True,
            output_type="ndarray",
        )
        
        # Dimensione del batch corrente
        batch_size = batch_features.shape[0]
        
        # Raccogliamo i percorsi delle immagini da questo batch
        batch_paths = []
        for i in range(len(batch)):
            try:
                # Recupera il percorso del file dall'oggetto batch
                file_path = batch.dataset.info.iloc[batch.indices[i]]["file_path"]
                batch_paths.append(file_path)
            except (AttributeError, IndexError, KeyError) as e:
                # Fallback nel caso in cui la struttura del dataset sia diversa
                batch_paths.append(f"unknown_batch_{batch_idx}_item_{i}")
                print(f"Impossibile recuperare il percorso per batch {batch_idx}, item {i}: {e}")
        
        # Verifica che il numero di percorsi corrisponda al numero di feature nel batch
        if len(batch_paths) != batch_size:
            print(f"ATTENZIONE: Mismatch tra numero di percorsi ({len(batch_paths)}) " +
                  f"e feature ({batch_size}) nel batch {batch_idx}")
            # Adatta le liste per far corrispondere le dimensioni
            if len(batch_paths) < batch_size:
                batch_paths.extend([f"generated_path_{batch_idx}_{i}" 
                                   for i in range(batch_size - len(batch_paths))])
            elif len(batch_paths) > batch_size:
                batch_paths = batch_paths[:batch_size]
        
        # Aggiungi i percorsi alla lista globale
        all_image_paths.extend(batch_paths)
        
        # Verifica che non superiamo la dimensione allocata
        if start_idx + batch_size <= total_samples:
            # Copia direttamente nel file memmap
            all_features[start_idx:start_idx+batch_size] = batch_features
            print_file_size(features_path)
            start_idx += batch_size
        else:
            print(f"ATTENZIONE: Superata dimensione allocata. Batch ignorato.")
        
        # Libera memoria immediatamente
        del batch_features
        torch.cuda.empty_cache()
        
        # Salva periodicamente i percorsi su disco
        if batch_idx % 10 == 0 or (args.test_mode and batch_idx == args.test_batches - 1):
            all_features.flush()
            print_file_size(features_path)
            # Salva i percorsi in un file JSON per il checkpoint
            temp_paths_file = os.path.join(run_path, "image_paths_temp.json")
            with open(temp_paths_file, 'w') as f:
                json.dump(all_image_paths, f)
            print(f"  Memoria sincronizzata. Progresso: {start_idx}/{total_samples} campioni")
            
    except Exception as e:
        print(f"Errore nell'elaborazione del batch {batch_idx}: {e}")
        import traceback
        traceback.print_exc()
        # Continua con il batch successivo

# Flush finale dei dati sul disco
all_features.flush()
print_file_size(features_path)
print(f"Completato. Salvate features per {start_idx}/{total_samples} campioni.")

# Modifica del file memmap per adattarlo al numero effettivo di campioni elaborati
if start_idx < total_samples:
    print(f"Ridimensionamento del file features da {total_samples} a {start_idx} campioni...")
    # Crea un nuovo file memmap con la dimensione corretta
    new_features_path = os.path.join(run_path, "features_resized.npy")
    new_features = np.memmap(new_features_path, dtype='float32', mode='w+', 
                             shape=(start_idx, feature_dim))
    # Copia i dati
    new_features[:] = all_features[:start_idx]
    new_features.flush()
    # Chiudi il vecchio file
    del all_features
    # Rinomina il nuovo file
    os.rename(new_features_path, features_path)
    print(f"File ridimensionato correttamente: {start_idx} campioni.")

# Salva i percorsi delle immagini in un file finale
print(f"Salvataggio dei {len(all_image_paths)} percorsi di immagini...")
# Assicuriamoci che la lunghezza dei percorsi corrisponda alle features
if len(all_image_paths) > start_idx:
    all_image_paths = all_image_paths[:start_idx]
    print(f"Percorsi troncati a {start_idx} elementi per corrispondere alle features.")
elif len(all_image_paths) < start_idx:
    all_image_paths.extend([f"missing_path_{i}" for i in range(start_idx - len(all_image_paths))])
    print(f"Aggiunti {start_idx - len(all_image_paths)} percorsi mancanti.")

with open(paths_file, 'w') as f:
    json.dump(all_image_paths, f)

# Verifica integrità del file features
try:
    test_load = np.memmap(features_path, dtype='float32', mode='r', 
                          shape=(start_idx, feature_dim))
    print(f"Verifica file features: OK. Dimensione finale: {test_load.shape}")
    del test_load
except Exception as e:
    print(f"Errore nella verifica del file features: {e}")

# Aggiunta di metadati sulle features al file di metadata
metadata["feature_count"] = start_idx
metadata["feature_dimension"] = feature_dim
metadata["features_path"] = features_path
metadata["image_paths_file"] = paths_file
metadata["processed_batches"] = batch_count if not args.test_mode else args.test_batches

with open(metadata_file_path, 'w') as f:
    json.dump(metadata, f, indent=2)

# Riepilogo finale
print("\n==== RIEPILOGO DELL'ELABORAZIONE ====")
print(f"Modalità: {'TEST' if args.test_mode else 'PRODUZIONE'}")
print(f"Batch elaborati: {batch_count if not args.test_mode else args.test_batches}")
print(f"Campioni totali estratti: {start_idx}")
print(f"Dimensione delle features: {feature_dim}")
print(f"File features: {features_path}")
print(f"File percorsi: {paths_file}")
print(f"File metadati: {metadata_file_path}")
print("====================================\n")