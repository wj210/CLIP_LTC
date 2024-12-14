import os
import os.path
from typing import Any, Callable, Optional, Tuple
from PIL import Image
import numpy as np
from collections import defaultdict
from torchvision.datasets import VisionDataset
import torch

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class CounterAnimal(VisionDataset):
    def __init__(
        self,
        root: str = '../imagenet/counteranimal',
        split: str = None,
        loader: Callable[[str], Any] = pil_loader,
        transform: Optional[Callable] = None,
        return_filename= False,
        diff_lvl = 'counter' # either common or counter
    ) -> None:
        super().__init__(root, transform=transform)
    
        self.loader = loader
        self.split = split
        random_generator = np.random.RandomState(42)
        assert diff_lvl in ['common','counter'], f"diff_lvl should be either common or counter, got {diff_lvl}"
        val_img,val_cls = [],[]
        test_img,test_cls = [],[]
        val_size = 16

        for cls_name in os.listdir(root):
            if '.txt' in cls_name or '.py' in cls_name:
                continue
            label = int(cls_name.split(' ')[0])
            diff_types = os.listdir(os.path.join(root,cls_name))
            for diff_type in diff_types:
                if diff_lvl in diff_type:
                    diff_dir = diff_type
                    break
            cls_path = os.path.join(root,cls_name,diff_dir)
            cls_files = os.listdir(cls_path)
            num_cls_files = len(cls_files)
            val_pos = random_generator.choice(np.arange(num_cls_files),val_size,replace=False).tolist()
            val_files = [os.path.join(cls_path,cls_files[j]) for j in val_pos]
            val_img.extend(val_files)
            test_pos = [i for i in range(num_cls_files) if i not in val_pos]
            test_files = [os.path.join(cls_path,cls_files[j]) for j in test_pos]
            test_img.extend(test_files)
            val_cls.extend([label]*val_size)
            test_cls.extend([label]*len(test_pos))

        val_cls, test_cls = torch.tensor(val_cls), torch.tensor(test_cls)
        self.return_filename = return_filename

        if split == 'val':
            self.samples = list(zip(val_img,val_cls))
        elif split == 'test':
            self.samples = list(zip(test_img,test_cls))
        elif split == 'all':
            self.samples = list(zip(val_img+test_img,torch.cat([val_cls,test_cls])))
        else:
            raise ValueError(f"split should be either val or test, got {split}")

        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_filename:
            return sample, (target,path)
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)
