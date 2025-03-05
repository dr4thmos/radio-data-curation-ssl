import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from collections import OrderedDict
from thingsvision import get_extractor_from_model
from torchvision.models import resnet18#, resnet50
#from thingsvision.utils.data import ImageDataset
from thingsvision.utils.data import DataLoader
from thingsvision.utils.storing import save_features
from data_classes import CustomUnlabeledDatasetWithPath
import torchvision.transforms as T
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#ckpt_path = "/home/tcecconello/radioimgs/radio-data-curation-ssl/model_weights/uja9qvb7/byol_hulk_aug_minmax_model_resnet18-uja9qvb7-ep=100.ckpt"
ckpt_path = "/home/tcecconello/radioimgs/radio-data-curation-ssl/model_weights/uja9qvb7.ckpt"
backbone_architecture = "resnet18"

model = resnet18(weights=None)
#print(model)
model.fc = torch.nn.Identity()
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False) # checkpoint = torch.load(ckpt_path, map_location={'cuda:0': device})


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

#salvare features (probabilente customizzare dataloader)

data_path='/home/tcecconello/radioimgs/radio-data-curation-ssl/LoTSS/cutouts'
batch_size = 16

transforms = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x: x.repeat(3,1,1)),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # stretch -1 1
    T.Resize(224)
])


dataset = CustomUnlabeledDatasetWithPath(
    data_path=data_path,
    loader_type="npy",
    transforms=transforms
    #extractor.get_transformations(resize_dim=256, crop_dim=224),
)

batches = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    #num_workers=0,
    backend=extractor.get_backend() # backend framework of model
)


#module_name = 'visual' avgpool
module_name = 'avgpool'

features = extractor.extract_features(
    batches=batches,
    module_name=module_name,
    flatten_acts=True,
    output_type="ndarray", # or "tensor" (only applicable to PyTorch models of which CLIP and DINO are ones!)
)
 
save_features(features, out_path='/home/tcecconello/radioimgs/radio-data-curation-ssl/features_byol_hulk_norm_minmax', file_format='npy') # file_format can be set to "npy", "txt", "mat", "pt", or "hdf5"
