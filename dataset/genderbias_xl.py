import os
import os.path
from typing import Any, Callable, Optional, Tuple
from PIL import Image
import pandas as pd
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset
import torch
from collections import defaultdict
import json
import numpy as np
import random

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

val_occ = ['Supervisor of personal care and service workers', 'Printing press operator', 'Public relations specialist', 'Fast food and counter worker', 'Logistician', 'Artist', 'Food service manager', 'Translator', 'Loan officer', 'Ophthalmic medical technician', 'Paralegal', 'Psychiatric technician', 'Medical assistant', 'Supervisor of transportation and material moving workers', 'First-line supervisor of office and administrative support workers', 'Computer hardware engineer', 'Management analyst', 'Accountant', 'Operations research analyst', 'Computer numerically controlled tool operator', 'Phlebotomist', 'Secondary school teacher', 'Entertainment and recreation manager', 'Social and community service manager', 'Occupational health and safety specialist']

occupations = ['Correctional officer', 'Biological scientist', 'Software developer', 'Licensed practical nurse', 'Chief executive', 'Psychiatric technician', 'Legal secretary', 'Housekeeping cleaner', 'Probation officer', 'Medical appliance technician', 'Web and digital interface designer', 'Postsecondary teacher', 'Software quality tester', 'First-line supervisor of police and detectives', 'Laundry worker', 'Loan interviewer', 'Information security analyst', 'Secondary school teacher', 'Counter and rental clerk', 'First-line supervisor of production and operating workers', 'Computer numerically controlled tool operator', 'School psychologist', 'Elementary and middle school teacher', 'Artist', 'Cost estimator', 'Receptionist', 'Computer programmer', 'Electrical and electronics engineer', 'Insurance claims and policy processing clerk', 'Lawyer', 'Building cleaner', 'Administrative services manager', 'Dental hygienist', 'Waitress', 'Purchasing manager', 'Electrical and electronic engineering technician', 'Physician assistant', 'Logistician', 'First-line supervisor of construction trades and extraction workers', 'Construction and building inspector', 'Mechanical engineer', 'First-line supervisor of retail sales workers', 'Database administrator', 'Materials engineer', 'Transportation service attendant', 'Customer service representative', 'Pharmacy technician', 'Editor', 'First-line supervisor of correctional officers', 'Massage therapist', 'Miscellaneous health technician', 'Chef', 'Civil engineer', 'Social and human service assistant', 'First-line supervisor of office and administrative support workers', 'Advertising sales agent', 'Waiter', 'Dentist', 'Telecommunication equipment installer', 'Property assessor', 'Medical and health services manager', 'Project management specialist', 'Entertainment and recreation manager', 'Security guard', 'Training and development specialist', 'Police officer', 'Emergency medical technician', 'Surgical technologist', 'File Clerk', 'Web developer', 'First-line supervisor of non-retail sales workers', 'Translator', 'Travel agent', 'Marketing specialist', 'Real estate broker', 'Financial and investment analyst', 'Public relations specialist', 'Computer network architect', 'Physical therapist assistant', 'Social and community service manager', 'Parts salesperson', 'Mental health counselor', 'Author', 'Industrial production manager', 'Supervisor of transportation and material moving workers', 'Lodging manager', 'Food service manager', 'Payroll and timekeeping clerk', 'Pharmacist', 'Accountant', 'Personal financial advisor', 'Special education teacher', 'Public relations manager', 'Refractory machinery mechanic', 'Veterinary technician', 'Insurance sales agent', 'Chemical technician', 'Postal service mail carrier', 'Insurance underwriter', 'Compliance officer', 'Network and computer systems administrator', 'Retail salesperson', 'Medical assistant', 'Healthcare social worker', 'Construction manager', 'First-line supervisor of security workers', 'Chiropractor', 'Computer support specialist', 'Computer systems analyst', 'Radiologic technician', 'Medical records specialist', 'Loan officer', 'Computer systems manager', 'Sales manager', 'Boiler operator', 'Order clerk', 'Computer hardware engineer', 'Paralegal', 'Architectural and engineering manager', 'Aerospace engineer', 'Actor', 'Executive secretary', 'Marketing manager', 'Fast food and counter worker', 'Baker', 'Chemical engineer', 'Wastewater treatment system operator', 'Dishwasher', 'Nutritionist', 'Ophthalmic medical technician', 'Interior designer', 'Tax preparer', 'Graphic designer', 'Photographer', 'First-line supervisor of housekeeping and janitorial workers', 'Physical therapist', 'Operation manager', 'First-line supervisor of food preparation and serving workers', 'Flight attendant', 'Human resources manager', 'Medical scientist', 'Cook', 'Printing press operator', 'Painting worker', 'Food preparation worker', 'Cutting worker', 'Public safety telecommunicator', 'Painter', 'Training and development manager', 'Operations research analyst', 'Billing and posting clerk', 'Tutor', 'Sewing machine operator', 'Postal service clerk', 'Director', 'Facilities manager', 'Supervisor of personal care and service workers', 'Chemists and materials scientist', 'Occupational health and safety specialist', 'Human resources worker', 'Respiratory therapist', 'Teaching assistant', 'Nurse practitioner', 'Pest control worker', 'Cafeteria attendant', 'Financial manager', 'Stocker', 'Management analyst', 'Food batchmaker', 'Nursing assistant', 'Surgeon', 'Dental assistant', 'Automotive service mechanic', 'Aircraft pilot', 'Filling machine operator', 'Phlebotomist', 'Fundraiser', 'Cashier']

separate_cls = ['Filling machine operator', 'Author', 'Aerospace engineer', 'Nutritionist', 'Software developer', 'Billing and posting clerk', 'Network and computer systems administrator', 'Executive secretary', 'Chef', 'Public relations manager', 'Physical therapist', 'Transportation service attendant', 'Marketing specialist', 'Operation manager', 'Web and digital interface designer', 'Property assessor', 'Tutor', 'Pharmacist', 'Building cleaner', 'Physical therapist assistant', 'Flight attendant', 'Mechanical engineer', 'First-line supervisor of housekeeping and janitorial workers', 'Computer systems manager', 'Order clerk']

def get_occupation_pairing():
    occupation_pairing = {}
    occupation_gender = {}
    male_ratio = {}
    occ_pairing_df = pd.read_csv('dataset/genderbias_occ.csv')
    for gender_type in ['job_male','job_female']:
        if gender_type == 'job_male':
            counterpart_col = 'job_female_ratio'
            counterpart_job = 'job_female'
            gender_ = 'male'
        else:
            counterpart_col = 'job_male_ratio'
            counterpart_job = 'job_male'
            gender_ = 'female'
        for job in occ_pairing_df[gender_type].unique():
            similar_rows = occ_pairing_df[occ_pairing_df[gender_type] == job]
            if gender_ == 'male':
                max_counterpart = similar_rows[counterpart_col].idxmax() # find the highest female workforce ratio
                ## get the workforce ratio
                male_workforce_ratio = 100 - similar_rows.loc[max_counterpart,'job_male_ratio']
            else:
                max_counterpart = similar_rows[counterpart_col].idxmin() # male role is actually 1-val, thus find min
                male_workforce_ratio = 100 - similar_rows.loc[max_counterpart,'job_female_ratio']
            
            other_job = occ_pairing_df.loc[max_counterpart,counterpart_job]
            occupation_pairing[job] = other_job
            occupation_gender[job] = gender_
            male_ratio[job] = male_workforce_ratio
    return occupation_pairing,occupation_gender,male_ratio


class Genderbias_xl(VisionDataset):
    def __init__(
        self,
        root: str = '../imagenet/genderbias_xl',
        split: str = 'val',
        loader: Callable[[str], Any] = pil_loader,
        transform: Optional[Callable] = None,
        return_filename= False,
    ) -> None:
        super().__init__(root, transform=transform)
        self.loader = loader
        self.split = split
        self.imgs = []
        self.cls_idx = [] # points to the index of the classnames of female or male 0 for female 1 for male
        self.cls = []

        self.occupations = occupations
        new_occupations = []

        ## Filter out occ without female and male
        for occ in self.occupations:
            occ_path = os.path.join(root,occ.replace(' ','_'))
            if not os.path.exists(os.path.join(occ_path,'female')) or not os.path.exists(os.path.join(occ_path,'male')): # make sure have both genders
                continue
            new_occupations.append(occ)
        self.occupations = new_occupations
        print (f'Occupations: {len(self.occupations)}')

        test_occ = [occ for occ in self.occupations if occ not in val_occ]

        ## Get occupation pairing
        occupation_pairing,occupation_gender,_ = get_occupation_pairing()

        split_occ = val_occ if split == 'val' else test_occ
        self.img = []
        self.gender = []
        self.cls = []
        self.cls_idx = []
        self.occs = []
        gender_map = {'female':0,'male':1}
        for occ in split_occ:
            for occ_g in ['male','female']:
                img_files = os.listdir(os.path.join(root,occ.replace(' ','_'),occ_g))
                self.img.extend([os.path.join(root,occ.replace(' ','_'),occ_g,f) for f in img_files])
                self.gender.extend([gender_map[occ_g]]*len(img_files))
                other_occ = occupation_pairing[occ]
                self.cls_idx.extend([[occupations.index(oo) for oo in [occ,other_occ]] for _ in img_files])
                self.occs.extend([occupations.index(occ)]*len(img_files))
                self.cls.extend([0]*len(img_files))

        self.return_filename = return_filename
        self.cls = torch.tensor(self.cls)
        self.gender = torch.tensor(self.gender)

        self.samples = [(self.img[i],self.cls[i],self.gender[i],self.cls_idx[i],self.occs[i]) for i in range(len(self.img))]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target,gender,cls_idx,occ = self.samples[index]
        cls_idx = torch.tensor(cls_idx)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_filename:
            return sample, (target, gender,cls_idx,occ,path)
        return sample, (target,gender,cls_idx,occ)

    def __len__(self) -> int:
        return len(self.samples)


class FilteredGB(Dataset):
    def __init__(self, original_dataset,num=500):
        selected_occ = set(['Lawyer','Dentist','Chief executive','Transportation service attendant','Building cleaner','Web developer','Receptionist','Civil engineer','Mechanical engineer','Nutritionist'])

        self.original_dataset = original_dataset
        self.filtered_indices = [
            i for i in range(len(original_dataset))
            if occupations[original_dataset[i][1][-2]] in selected_occ  # Filter condition: 2nd item is 1 or 2
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
        


