import os
import argparse
import json

def parse_model_parameters(param_list):
    """Parsa i parametri del modello da una lista di stringhe key=value."""
    return {kv.split('=')[0]: kv.split('=')[1] for kv in param_list if '=' in kv}

parser = argparse.ArgumentParser(description="Parametrized script for feature extraction.")

parser.add_argument('--cuda_devices', type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "4"),
                    help="CUDA device(s) to use in the format \"0,2,..\"")
parser.add_argument('--features_folder', type=str, default=os.path.join(os.getcwd(), 'features'),
                    help="Path to features folder.")
parser.add_argument('--model_name', type=str, default='OpenCLIP', help="Model name.")
parser.add_argument('--model_parameters', type=str, nargs='*', default=['variant=ViT-B-32', 'dataset=laion400m_e32'],
                    help="Model parameters as key=value pairs.")
parser.add_argument('--backend', type=str, default='pt', help="Backend to use.")
parser.add_argument('--source', type=str, default='custom', help="Data source type.")
parser.add_argument('--batch_size', type=int, default=16, help="Batch size for processing.")
parser.add_argument('--data_path', type=str, default=os.path.join(os.getcwd(), 'LoTSS/cutouts'),
                    help="Path to input data.")
parser.add_argument('--data_selection', type=str, default='info_test.json',
                    help="File .json with selection of data to be used")
parser.add_argument('--module_name', type=str, default='visual', help="Module name.")
parser.add_argument('--normalization', type=str, default='imagenet', help="which norm to use")
parser.add_argument('--resize', type=int, default=224, help="Image dimensional resize.")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
model_parameters = parse_model_parameters(args.model_parameters)

#model_parameters = {'variant': args.model_variant, 'dataset': args.model_dataset}
#variant = build_variant(model_parameters)

print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"Features Folder: {args.features_folder}")
print(f"Model: {args.model_name}")
print(f"Backend: {args.backend}")
print(f"Source: {args.source}")
print(f"Batch Size: {args.batch_size}")
print(f"Data Path: {args.data_path}")
print(f"Data Selection: {args.data_selection}")
print(f"Module Name: {args.module_name}")
print(f"Model Parameters: {model_parameters}")


from collections import OrderedDict
from thingsvision import get_extractor_from_model, get_extractor
from torchvision.models import resnet18#, resnet50
#from thingsvision.utils.data import ImageDataset
from thingsvision.utils.data import DataLoader
from thingsvision.utils.storing import save_features
from data_classes import CustomUnlabeledDatasetWithPath
import torchvision.transforms as T
import torch
import re

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # stretch -1 1
if args.normalization == "imagenet":
    norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # stretch -1 1
if args.normalization == "other":
    norm = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # stretch -1 1

transforms = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x: x.repeat(3,1,1)),
    norm,
    T.Resize(args.resize)
])


dataset = CustomUnlabeledDatasetWithPath(
    data_path=args.data_path,
    loader_type="npy",
    transforms=transforms,
    datalist=args.data_selection
    #extractor.get_transformations(resize_dim=256, crop_dim=224),
)

batches = DataLoader(
    dataset=dataset,
    batch_size=args.batch_size,
    #num_workers=0,
    backend=extractor.get_backend() # backend framework of model
)

features = extractor.extract_features(
    batches=batches,
    module_name=args.module_name,
    flatten_acts=True,
    output_type="ndarray", # or "tensor" (only applicable to PyTorch models of which CLIP and DINO are ones!)
)
 
save_features(features, out_path=out_path, file_format='npy') # file_format can be set to "npy", "txt", "mat", "pt", or "hdf5"

metadata = {}
metadata["model_name"]          = args.model_name
metadata["model_parameters"]    = model_parameters
metadata["backend"]             = args.backend
metadata["source"]              = args.source
metadata["data_path"]           = args.data_path
metadata["data_selection"]      = args.data_selection
metadata["module_name"]         = args.module_name
metadata["normalization"]       = args.normalization
metadata["resize"]              = args.resize
# Salvare json con modello e dataset utilizzato.
# Salvataggio su file JSON
with open(os.path.join(out_path,"metadata.json"), "w") as file:
    json.dump(metadata, file, indent=4)  # indent=4 per leggibilità