import random
from json import load
import os
import math
import numpy as np
import pandas as pd

import torch
from monai import data

from packaging import version
_persistent_workers = False if version.parse(torch.__version__) < version.parse('1.8.2') else True

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank:self.total_size:self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[:(self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0,high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


import os
from glob import glob
import numpy as np
import json
import random
import nibabel as nib
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.multiprocessing import Pool


def label_transform(label):
    patient_id, death_duration_month, death_event = label
    patient_id = str(patient_id)
    death_duration_month = float(death_duration_month)
    death_event = bool(death_event)
    
    return {'patient_id': patient_id, 'death_duration_month': death_duration_month, 'death_event': death_event}


class MRIs(Dataset):
    def __init__(self, args, transform, phase):
        super().__init__()
        self.dataroot = args.dataroot
        self.transform = transform
        self.phase = phase

        label_path = os.path.join(self.dataroot, f'{self.phase}_label.csv')
        df = pd.read_csv(label_path)

        df = df[['patient_id', 'death_duration_month', 'death_event']]
        self.label_dict = {}
        for i in range(len(df)):
            label = label_transform(df.iloc[i])
            patient_id = label['patient_id']
            self.label_dict[f'{patient_id}'] = label
        
        self.dataset_paths = glob(os.path.join(self.dataroot, f'{self.phase}', 'preprocessed', '*'))
        self.event_dataset_paths = [data for data in self.dataset_paths if self.label_dict[f'{data.split("/")[-1]}']['death_event']]
        
        print(f'Total {self.phase} dataset size - patient: {len(list(set([p[-1] for p in self.dataset_paths])))}, volume: {len(self.dataset_paths)}')
    
    def get_event_data(self):
        patient_dir = random.choice(self.event_dataset_paths)
        t1, t1ce, t2, flair, mask, tumor_mask, patient_id = self._single_scan(patient_dir)
        
        data = self.transform({
            't1': t1, 
            't1ce': t1ce, 
            't2': t2, 
            'flair': flair, 
            'mask': mask, 
            'tumor_mask': tumor_mask
        })
        
        label = self.label_dict[f'{patient_id}']
        label = {k: v if type(v) == str else torch.tensor(v) for k, v in label.items()}
        data.update(label)
        
        return data
        
    def __getitem__(self, index):
        patient_dir = self.dataset_paths[index]
        t1, t1ce, t2, flair, mask, tumor_mask, patient_id = self._single_scan(patient_dir)
        
        data = self.transform({
            't1': t1, 
            't1ce': t1ce, 
            't2': t2, 
            'flair': flair, 
            'mask': mask, 
            'tumor_mask': tumor_mask, 
            'patient_id': patient_id
        })
        
        label = self.label_dict[f'{patient_id}']
        label = {k: v if type(v) == str else torch.tensor(v) for k, v in label.items()}
        data.update(label)
        
        return data

    def __len__(self):
        return len(self.dataset_paths)
        
    def _single_scan(self, patient_dir):
        patient_id = patient_dir.split('/')[-1]
        t1_path = os.path.join(patient_dir, 't1.nii.gz')
        t1ce_path = os.path.join(patient_dir, 't1ce.nii.gz')
        t2_path = os.path.join(patient_dir, 't2.nii.gz')
        flair_path = os.path.join(patient_dir, 'flair.nii.gz')
        mask_path = os.path.join(patient_dir, 'brain_mask.nii.gz')
        tumor_mask_path = os.path.join(patient_dir, 'tumor_mask.nii.gz')
        
        t1 = nib.load(t1_path).get_fdata()
        t1ce = nib.load(t1ce_path).get_fdata()
        t2 = nib.load(t2_path).get_fdata()
        flair = nib.load(flair_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        tumor_mask = nib.load(tumor_mask_path).get_fdata()
        
        return [t1, t1ce, t2, flair, mask, tumor_mask, patient_id]


class MRIsTrain(MRIs):
    def __init__(self, args, transform):
        super().__init__(args, transform=transform, phase='train')


class MRIsValidation(MRIs):
    def __init__(self, args, transform):
        super().__init__(args, transform=transform, phase='valid')
        

# get data loaders
def get_train_loader(args, batch_size, workers, train_transform=None):
    train_ds = MRIsTrain(args, transform=train_transform)
    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(train_ds,
                                    batch_size=batch_size,
                                    shuffle=(train_sampler is None),
                                    num_workers=workers,
                                    sampler=train_sampler,
                                    pin_memory=True,
                                    persistent_workers=True,
                                    prefetch_factor=8,
                                    drop_last=False)
    return train_loader


def get_val_loader(args, batch_size, workers, val_transform=None):
    val_ds = MRIsValidation(args, transform=val_transform)
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(val_ds,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=workers,
                                 sampler=val_sampler,
                                 pin_memory=True,
                                 persistent_workers=True,
                                 prefetch_factor=8,
                                 drop_last=False)
    return val_loader
