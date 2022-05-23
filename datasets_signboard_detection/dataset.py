import torch
from PIL import Image
from torch.utils.data import Dataset
import os 

class Labelizer():
    def __init__(self):
        super().__init__()
        self.labels = {'background': 0, 'bien': 1}
        self.inv_labels = {0: 'background', 1: 'bien'}
    def transform(self, label):
        return self.labels[label]
    
    def inverse_transform(self, ys):
        return self.inv_labels(ys)
    
    def num_classes(self):
        return len(self.labels)

class PoIDataset(Dataset):
    def __init__(self,
                 list_img,
                 data_path,
                 transforms=None):
        self.list_img = list_img
        self.data_path = data_path
        self.transforms = transforms
    
    def __len__(self):
        return len(self.list_img)
    
    def __getitem__(self, idx):
        img_name = self.list_img[idx]
        image = Image.open(os.path.join(self.data_path,img_name)).convert('RGB')
        target = {}
        if self.transforms is not None:
            image = self.transforms(image)
        return image, target