import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Callable
from PIL import Image

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class CelebA(Dataset):
    def __init__(self, root='../imagenet/celeba', split='train', loader:Callable[[str], Any] = pil_loader,transform=None,return_filename=False):
        self.root = root
        self.split = split
        self.loader = loader
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}

        with open(os.path.join(self.root, 'list_attr_celeba.txt'), 'r') as f:
            lines = f.readlines()
        attribute_names = lines[1].strip().split()  # Skip first placeholder in the header
        data = []
        for line in lines[2:]:  # Start from line 3 to skip header
            parts = line.strip().split()
            image_id = parts[0]
            attributes = list(map(int, parts[1:]))
            assert len(attributes) == len(attribute_names)
            data.append([image_id] + attributes)

        # Create a DataFrame from the data
        self.metadata_df = pd.DataFrame(data, columns=['image_id'] + attribute_names)

        # Read partition data from list_eval_partition.txt
        with open(os.path.join(self.root, 'list_eval_partition.txt'), 'r') as f:
            partition_data = [line.strip().split() for line in f.readlines()]
        
        # Convert partition data to a DataFrame
        self.split_df = pd.DataFrame(partition_data, columns=['image_id', 'partition'])
        self.split_df['partition'] = self.split_df['partition'].astype(int)

        # Merge partition info into the main metadata_df
        self.metadata_df = self.metadata_df.merge(self.split_df, on='image_id')
        self.metadata_df = self.metadata_df[self.metadata_df['partition'] == self.split_dict[self.split]]

        # Get the y values
        self.y_array = self.metadata_df['Blond_Hair'].values
        self.confounder_array = self.metadata_df['Male'].values
        self.y_array[self.y_array == -1] = 0
        self.confounder_array[self.confounder_array == -1] = 0
        self.group_array = (self.y_array * 2 + self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['image_id'].values
        self.split_array = self.metadata_df['partition'].values

        self.targets = torch.tensor(self.y_array)
        self.targets_group = torch.tensor(self.group_array)
        self.targets_spurious = torch.tensor(self.confounder_array)

        self.target_name = {0: 'dark-haired',1: 'blond'}
        self.group_name = {0: 'non_blond_female', 1: 'non_blond_male', 2: 'blond_female', 3: 'blond_male'}
        self.class_counts = {}
        for tk in self.target_name.keys():
            self.class_counts[tk] = (self.y_array == tk).sum().item()
        self.group_counts = {}
        for gk in self.group_name.keys():
            self.group_counts[gk] = (self.group_array == gk).sum().item()
        
        for k,v in self.group_counts.items():
            print(f'Group {k} count: {v}')
        self.transform = transform
        self.return_filename = return_filename


    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        img_filename = os.path.join(self.root, 'img_align_celeba', self.filename_array[idx])
        img = self.loader(img_filename)
        x = self.transform(img) if self.transform else img

        y = self.targets[idx]
        y_group = self.targets_group[idx]
        path = self.filename_array[idx]
        if self.return_filename:
            return x, (y, y_group,img_filename) 
        return x, (y, y_group)


