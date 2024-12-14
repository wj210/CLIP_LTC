import os
import os.path
from typing import Any, Callable, Optional, Tuple
from PIL import Image
import pandas as pd
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset
import torch
from dataset.cub_classes import places365,place_classes


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class BinaryWaterbirds(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str,
        loader: Callable[[str], Any] = pil_loader,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_filename= False,
        return_spurious = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
    
        self.loader = loader
        csv = pd.read_csv(os.path.join(root, 'metadata.csv'))
        split = {'test': 2, 'valid': 1, 'train': 0}[split]
        csv = csv[csv['split'] == split]
        self.y_array = csv['y'].values
        self.confounder_array = csv['place'].values
        self.group_array = (self.y_array * 2 + self.confounder_array).astype('int')
        self.places = csv['place_filename'].values
        self.places = [p.split('/')[2] for p in self.places]
        self.places = [place_classes.index(p) for p in self.places]

        self.targets = torch.tensor(self.y_array)
        self.targets_group = torch.tensor(self.group_array)
        self.targets_spurious = torch.tensor(self.confounder_array)
        self.places = torch.tensor(self.places)

        self.return_spurious = return_spurious

        if not return_spurious:
            self.samples = [(os.path.join(root, csv.iloc[i]['img_filename']), csv.iloc[i]['y'],self.targets_group[i],self.places[i]) for i in range(len(csv))]

        else:
            self.samples = [(os.path.join(root, csv.iloc[i]['img_filename']), csv.iloc[i]['y'],self.targets_group[i],self.targets_spurious[i],self.places[i]) for i in range(len(csv))]



        self.target_name = {0: 'landbird',1: 'waterbird'}
        self.group_name = {0: 'landbird_land', 1: 'landbird_water', 2: 'waterbird_land', 3: 'waterbird_water'}
        self.class_counts = {}
        for tk in self.target_name.keys():
            self.class_counts[tk] = (self.y_array == tk).sum().item()
        self.group_counts = {}
        for gk in self.group_name.keys():
            self.group_counts[gk] = (self.group_array == gk).sum().item()
        
        for k,v in self.group_counts.items():
            print(f'Group {k} count: {v}')
        
        self.return_filename = return_filename
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if not self.return_spurious:
            path, target,group,place_name = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            if self.return_filename:
                return sample, (target, group,place_name, path)
            return sample, (target, group,place_name)
        else:
            path, target,group,spurious,place_name = self.samples[index]
            sample = self.transform(self.loader(path))
            return sample, (target, group, spurious, place_name)

    def __len__(self) -> int:
        return len(self.samples)


class FilteredWB(Dataset):
    def __init__(self, original_dataset,num=500):
        """
        Args:
            original_dataset (Dataset): The original dataset to filter.
        """
        self.original_dataset = original_dataset
        self.filtered_indices = [
            i for i in range(len(original_dataset))
            if original_dataset[i][1][0].item() in [1, 2]  # Filter condition: 2nd item is 1 or 2
        ][:num]

    def __getitem__(self, index):
        """
        Get the filtered sample by index.
        """
        original_index = self.filtered_indices[index]
        return self.original_dataset[original_index]

    def __len__(self):
        """
        Length of the filtered dataset.
        """
        return len(self.filtered_indices)