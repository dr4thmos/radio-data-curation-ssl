import os
import os.path
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from image_loaders import load_fits_image, load_npy_image, load_npy_to_squeeze_image ,load_grey_image, load_rgb_image

class CustomDataset(Dataset, ABC):
    def __init__(self, data_path: str, loader_type: str, transforms, datalist='info.json'):
        self.data_path = Path(data_path)
        self.transform = transforms
        self.info = pd.read_json(os.path.join(self.data_path, datalist),  orient="index")
        self.loaders = {
            'fits': load_fits_image,
            'npy': load_npy_image,
            'npy_to_squeeze': load_npy_to_squeeze_image,
            'grey_image': load_grey_image,
            'rgb_image': load_rgb_image
        }
        if loader_type in self.loaders:
            self.load_image = self.loaders[loader_type]
        else:
            raise ValueError(f"Loader type '{loader_type}' is not supported. Available loaders: {list(self.loaders.keys())}")

    @abstractmethod
    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.info)
    
class CustomUnlabeledDatasetWithPath(CustomDataset):
    def __getitem__(self, index):
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(os.path.join(self.data_path, self.info.iloc[index]["file_path"]))
        return self.transform(img)#, -1, os.path.join(self.data_path, self.info.iloc[index]["target_path"])

class CustomLabeledDatasetWithPath(CustomDataset):
    def __init__(self, data_path: str, loader_type: str, transforms, datalist='info.json'):
        super().__init__(data_path=data_path, loader_type=loader_type, transforms=transforms, datalist=datalist)
        self.class_to_idx = self.enumerate_classes()
        self.class_names = self.class_name_list()
        self.classes = self.class_names
        self.num_classes = len(self.class_to_idx)

    def class_name_list(self):
        return [cls_name for cls_name in self.info["source_type"].unique()]
    
    def enumerate_classes(self):
        return {cls_name: i for i, cls_name in enumerate(self.info["source_type"].unique())}

    def __getitem__(self, index):
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(os.path.join(self.data_path, self.info.iloc[index]["target_path"]))
        label = self.info.iloc[index]["source_type"]
        return self.transform(img), self.class_to_idx[label], os.path.join(self.data_path, self.info.iloc[index]["target_path"])

