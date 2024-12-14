import os
import os.path
from typing import Any, Callable, Optional, Tuple
from PIL import Image
import pandas as pd
from torchvision.datasets import VisionDataset
import torch
from collections import defaultdict

def fairface_template(x,grp):
    if grp == 'age':
        if '-' in x:
            before,after = x.split('-')
            return f'a photo of a person between {before} and {after} years old.'
        else:
            return f'a photo of a person {x} years old.'
    elif grp == 'gender':
        return f'a photo of a {x.lower()} person.'
    elif grp == 'race':
        return f'a photo of a {x.replace("_"," ").lower()} person.'
    else:
        raise ValueError(f'grp {grp} not supported')

concepts = ['good', 'evil', 'smart', 'dumb', 'attractive', 'unattractive', 'lawful', 'criminal', 'friendly', 'unfriendly']
attr_cls = {'age':['0-2','3-9','10-19','20-29','30-39','40-49','50-59','60-69','more than 70'],
           'gender':['Male','Female'],
           'race':['White','Latino_Hispanic','Indian','East Asian','Black','Southeast Asian','Middle Eastern']}

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class Fairface(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str,
        loader: Callable[[str], Any] = pil_loader,
        transform: Optional[Callable] = None,
        return_filename= False,
    ) -> None:
        super().__init__(root, transform=transform)
    
        self.loader = loader
        csv = pd.read_csv(os.path.join(root, f'fairface_label_{split}.csv'))
        self.age = csv['age']
        self.gender = csv['gender']
        self.race = csv['race']
        self.cls_counts = {}
        for grp in ['age','gender','race']:
            for clsname in attr_cls[grp]:
                self.cls_counts[clsname] = csv[grp].value_counts().get(clsname,0)

        ## map str to the pos of the clsnames
        self.age = torch.tensor(self.age.map({k:v for v,k in enumerate(attr_cls['age'])}))
        self.gender = torch.tensor(self.gender.map({k:v for v,k in enumerate(attr_cls['gender'])}))
        self.race = torch.tensor(self.race.map({k:v for v,k in enumerate(attr_cls['race'])}))

        
        for k,v in self.cls_counts.items():
            print(f'cls {k} count: {v}')
        self.return_filename = return_filename

        self.samples = [(os.path.join(root,csv.iloc[i]['file']), self.age[i], self.gender[i],self.race[i]) for i in range(len(csv))]
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, age,gender,race = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_filename:
            return sample, (age,gender,race, path)
        return sample, (age,gender,race)

    def __len__(self) -> int:
        return len(self.samples)


def max_skewness(attns,mlps,attr_labels,classifier,k=1000,grp_types = ['gender']):
    out_skew = defaultdict(list)
    for grp_type in grp_types:
        activations = attns.sum(dim=(1,2)) + mlps.sum(dim=1)
        scores = (activations @ classifier).float().T
        for score in scores:
            top_k_pos = score.topk(k).indices
            grp_label = attr_labels[grp_type][top_k_pos] 
            grp_counts = torch.bincount(grp_label,minlength = len(attr_cls[grp_type]))
            r_ratio = grp_counts/k
            skew = torch.log(r_ratio/(1/len(attr_cls[grp_type])))
            out_skew[grp_type].append(skew.max().item())
    out_skew = {k:sum(v)/len(v) for k,v in out_skew.items()}
    return out_skew
        




