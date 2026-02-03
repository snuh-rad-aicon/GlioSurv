from re import L
import random
import numpy as np
import torch
from monai import transforms


class LastTransfromMRIs(transforms.MapTransform):
    def __init__(
        self,
        keys = ("t1", "t1ce", "t2", "flair", "mask"),
        gaussian_noise: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.gaussian_noise = gaussian_noise

    def __call__(self, data):
        d = dict(data)
        
        t1 = d['t1']
        t1ce = d['t1ce']
        t2 = d['t2']
        flair = d['flair']
        mask = d['mask']
        
        t1_signal = t1[mask > 0]
        t1ce_signal = t1ce[mask > 0]
        t2_signal = t2[mask > 0]
        flair_signal = flair[mask > 0]
        meanv_t1, stdv_t1 = t1_signal.mean(), t1_signal.std()
        meanv_t1ce, stdv_t1ce = t1ce_signal.mean(), t1ce_signal.std()
        meanv_t2, stdv_t2 = t2_signal.mean(), t2_signal.std()
        meanv_flair, stdv_flair = flair_signal.mean(), flair_signal.std()
        
        t1 = (t1 - meanv_t1) / (stdv_t1 + 1e-8)
        t1ce = (t1ce - meanv_t1ce) / (stdv_t1ce + 1e-8)
        t2 = (t2 - meanv_t2) / (stdv_t2 + 1e-8)
        flair = (flair - meanv_flair) / (stdv_flair + 1e-8)
        
        mris = torch.cat([t1, t1ce, t2, flair], dim=0)
        if self.gaussian_noise:
            prob = random.random()
            if prob < 0.9:
                mris = mris + torch.randn_like(mris) * 0.05
                
        mris = mris * mask
        # mris = mris.clamp(-2.5, 2.5)
        
        d['input'] = mris
        
        if torch.isnan(mris).any():
            patient_id = d['patient_id']
            print("NaN values present in mris of patient: ", patient_id)
            patient_id = d['asd']
        
        return d


def get_MAE_train_transforms():
    train_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair", "mask", "tumor_mask"], channel_dim='no_channel'),
            transforms.RandFlipd(keys=["t1", "t1ce", "t2", "flair", "mask", "tumor_mask"], prob=0.5, spatial_axis=0),
            transforms.RandAffined(
                keys=["t1", "t1ce", "t2", "flair", "mask", "tumor_mask"],
                mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest", "nearest"),
                prob=0.9,
                rotate_range=((-np.pi/12, np.pi/12), (-np.pi/12, np.pi/12), (-np.pi/24, np.pi/24)),
                translate_range=((-8, 8), (-8, 8), (-4, 4)),
                scale_range=((-0.001, 0.001), (-0.001, 0.001), (-0.0005, 0.0005)),
                padding_mode="border",
            ),
            transforms.RandAdjustContrastd(keys=["t1", "t1ce", "t2", "flair"], prob=0.15, gamma=(0.7, 1.5)),
            LastTransfromMRIs(gaussian_noise=False),
            transforms.ToTensord(keys=["input", "mask", "tumor_mask"], dtype=torch.float32),
        ]
    )
    return train_transform


def get_classification_train_transforms():
    train_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair", "mask", "tumor_mask"], channel_dim='no_channel'),
            transforms.RandFlipd(keys=["t1", "t1ce", "t2", "flair", "mask", "tumor_mask"], prob=0.5, spatial_axis=0),
            transforms.RandAffined(
                keys=["t1", "t1ce", "t2", "flair", "mask", "tumor_mask"],
                mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest", "nearest"),
                prob=0.9,
                rotate_range=((-np.pi/12, np.pi/12), (-np.pi/12, np.pi/12), (-np.pi/24, np.pi/24)),
                translate_range=((-8, 8), (-8, 8), (-4, 4)),
                scale_range=((-0.001, 0.001), (-0.001, 0.001), (-0.0005, 0.0005)),
                padding_mode="border",
            ),
            transforms.RandAdjustContrastd(keys=["t1", "t1ce", "t2", "flair"], prob=0.15, gamma=(0.9, 1.11)),
            LastTransfromMRIs(gaussian_noise=False),
            transforms.ToTensord(keys=["input", "mask", "tumor_mask"], dtype=torch.float32),
        ]
    )
    return train_transform
    
    
def get_val_transforms():
    val_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair", "mask", "tumor_mask"], channel_dim='no_channel'),
            LastTransfromMRIs(gaussian_noise=False),
            transforms.ToTensord(keys=["input", "mask", "tumor_mask"], dtype=torch.float32),
        ]
    )
    return val_transform
    