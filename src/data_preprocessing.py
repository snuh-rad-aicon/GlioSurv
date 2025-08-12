from multiprocessing import Pool, cpu_count
import os
from glob import glob
from tqdm import tqdm
import shutil
import ants
import pandas as pd
import argparse


iso_resolution = (1.0, 1.0, 1.0)
target_resolution = (1.0, 1.0, 4.0)
target_shape = [160, 192, 40]


def preprocessing_mri(patient_dir):
    patient_id = patient_dir.split('/')[-1]
    target_dir = os.path.join(target_root, patient_id)
    os.makedirs(target_dir, exist_ok=True)
    
    t1_path = os.path.join(patient_dir, f't1.nii.gz')
    t1ce_path = os.path.join(patient_dir, f't1ce.nii.gz')
    t2_path = os.path.join(patient_dir, f't2.nii.gz')
    flair_path = os.path.join(patient_dir, f'flair.nii.gz')
    
    t1_bet_path = os.path.join(patient_dir, f't1_bet.nii.gz')
    t1ce_bet_path = os.path.join(patient_dir, f't1ce_bet.nii.gz')
    t2_bet_path = os.path.join(patient_dir, f't2_bet.nii.gz')
    flair_bet_path = os.path.join(patient_dir, f'flair_bet.nii.gz')
    
    t1_bet_mask_path = os.path.join(patient_dir, f't1_bet_mask.nii.gz')
    t1ce_bet_mask_path = os.path.join(patient_dir, f't1ce_bet_mask.nii.gz')
    t2_bet_mask_path = os.path.join(patient_dir, f't2_bet_mask.nii.gz')
    flair_bet_mask_path = os.path.join(patient_dir, f'flair_bet_mask.nii.gz')
    
    os.system(f'mri_synthstrip -i {t1_path} -o {t1_bet_path} -m {t1_bet_mask_path}')
    os.system(f'mri_synthstrip -i {t1ce_path} -o {t1ce_bet_path} -m {t1ce_bet_mask_path}')
    os.system(f'mri_synthstrip -i {t2_path} -o {t2_bet_path} -m {t2_bet_mask_path}')
    os.system(f'mri_synthstrip -i {flair_path} -o {flair_bet_path} -m {flair_bet_mask_path}')
    
    os.system(f'robustfov -i {t1_path} -r {os.path.join(target_dir, "t1.nii.gz")}')
    brain_mask_path = os.path.join(target_dir, f'brain_mask.nii.gz')
    shutil.copy(t1_bet_mask_path, brain_mask_path)
    
    nifti_t1 = ants.image_read(os.path.join(target_dir, 't1.nii.gz'), reorient='RPI')
    nifti_t1ce = ants.image_read(t1ce_path, reorient='RPI')
    nifti_t2 = ants.image_read(t2_path, reorient='RPI')
    nifti_flair = ants.image_read(flair_path, reorient='RPI')
    
    nifti_t1ce_bet = ants.image_read(t1ce_bet_path, reorient='RPI')
    nifti_t2_bet = ants.image_read(t2_bet_path, reorient='RPI')
    nifti_flair_bet = ants.image_read(flair_bet_path, reorient='RPI')
    nifti_brain_mask = ants.image_read(brain_mask_path, reorient='RPI')
    
    nifti_t1 = ants.resample_image(nifti_t1, iso_resolution, use_voxels=False, interp_type=3)
    nifti_brain_mask = ants.resample_image_to_target(nifti_brain_mask, nifti_t1, interp_type=1)
    nifti_t1_bet = ants.mask_image(nifti_t1, nifti_brain_mask)
    
    nifti_t1ce_bet = ants.registration(fixed=nifti_t1_bet, moving=nifti_t1ce_bet, type_of_transform='Rigid')
    nifti_t1ce = ants.apply_transforms(fixed=nifti_t1ce_bet['warpedmovout'], moving=nifti_t1ce, transformlist=nifti_t1ce_bet['fwdtransforms'])
    nifti_t2_bet = ants.registration(fixed=nifti_t1_bet, moving=nifti_t2_bet, type_of_transform='Rigid')
    nifti_t2 = ants.apply_transforms(fixed=nifti_t2_bet['warpedmovout'], moving=nifti_t2, transformlist=nifti_t2_bet['fwdtransforms'])
    nifti_flair_bet = ants.registration(fixed=nifti_t1_bet, moving=nifti_flair_bet, type_of_transform='Rigid')
    nifti_flair = ants.apply_transforms(fixed=nifti_flair_bet['warpedmovout'], moving=nifti_flair, transformlist=nifti_flair_bet['fwdtransforms'])
    
    nifti_t1 = nifti_t1ce.new_image_like(nifti_t1.numpy())
    nifti_brain_mask = nifti_t1ce.new_image_like(nifti_brain_mask.numpy())
    nifti_t1_bet = ants.mask_image(nifti_t1, nifti_brain_mask)
    nifti_t1ce_bet = ants.mask_image(nifti_t1ce, nifti_brain_mask)
    nifti_t2_bet = ants.mask_image(nifti_t2, nifti_brain_mask)
    nifti_flair_bet = ants.mask_image(nifti_flair, nifti_brain_mask)
    
    nifti_t1.to_file(os.path.join(target_dir, 't1.nii.gz'))
    nifti_t1ce.to_file(os.path.join(target_dir, 't1ce.nii.gz'))
    nifti_t2.to_file(os.path.join(target_dir, 't2.nii.gz'))
    nifti_flair.to_file(os.path.join(target_dir, 'flair.nii.gz'))
    
    nifti_t1_bet.to_file(os.path.join(target_dir, 't1_bet.nii.gz'))
    nifti_t1ce_bet.to_file(os.path.join(target_dir, 't1ce_bet.nii.gz'))
    nifti_t2_bet.to_file(os.path.join(target_dir, 't2_bet.nii.gz'))
    nifti_flair_bet.to_file(os.path.join(target_dir, 'flair_bet.nii.gz'))
    nifti_brain_mask.to_file(os.path.join(target_dir, 'brain_mask.nii.gz'))

    os.system(f'hd_glio_predict -t1 {os.path.join(target_dir, "t1_bet.nii.gz")} -t1c {os.path.join(target_dir, "t1ce_bet.nii.gz")} -t2 {os.path.join(target_dir, "t2_bet.nii.gz")} -flair {os.path.join(target_dir, "flair_bet.nii.gz")} -o {os.path.join(target_dir, "tumor_mask.nii.gz")}')
    os.system(f'fslmaths {os.path.join(target_dir, "tumor_mask.nii.gz")} -mas {os.path.join(target_dir, "brain_mask.nii.gz")} {os.path.join(target_dir, "tumor_mask.nii.gz")}')
    os.system(f'fslmaths {os.path.join(target_dir, "tumor_mask.nii.gz")} -thr 1 -uthr 1 -bin {os.path.join(target_dir, "NE_tumor_mask.nii.gz")}')
    os.system(f'fslmaths {os.path.join(target_dir, "tumor_mask.nii.gz")} -thr 2 -uthr 2 -bin {os.path.join(target_dir, "CE_tumor_mask.nii.gz")}')
    os.system(f'fslmaths {os.path.join(target_dir, "tumor_mask.nii.gz")} -bin {os.path.join(target_dir, "WT_tumor_mask.nii.gz")}')

    t1_path = os.path.join(target_dir, f't1.nii.gz')
    t1ce_path = os.path.join(target_dir, f't1ce.nii.gz')
    t2_path = os.path.join(target_dir, f't2.nii.gz')
    flair_path = os.path.join(target_dir, f'flair.nii.gz')
    brain_mask_path = os.path.join(target_dir, f'brain_mask.nii.gz')
    tumor_mask_path = os.path.join(target_dir, f'tumor_mask.nii.gz')
    CE_tumor_mask_path = os.path.join(target_dir, f'CE_tumor_mask.nii.gz')
    NE_tumor_mask_path = os.path.join(target_dir, f'NE_tumor_mask.nii.gz')
    WT_tumor_mask_path = os.path.join(target_dir, f'WT_tumor_mask.nii.gz')
    
    nifti_t1 = ants.image_read(t1_path, reorient='RPI')
    nifti_t1ce = ants.image_read(t1ce_path, reorient='RPI')
    nifti_t2 = ants.image_read(t2_path, reorient='RPI')
    nifti_flair = ants.image_read(flair_path, reorient='RPI')
    nifti_brain_mask = ants.image_read(brain_mask_path, reorient='RPI')
    tumor_mask = ants.image_read(tumor_mask_path, reorient='RPI')
    CE_tumor_mask = ants.image_read(CE_tumor_mask_path, reorient='RPI')
    NE_tumor_mask = ants.image_read(NE_tumor_mask_path, reorient='RPI')
    WT_tumor_mask = ants.image_read(WT_tumor_mask_path, reorient='RPI')
    
    nifti_t1 = ants.resample_image(nifti_t1, target_resolution, use_voxels=False, interp_type=3)
    nifti_t1ce = ants.resample_image(nifti_t1ce, target_resolution, use_voxels=False, interp_type=3)
    nifti_t2 = ants.resample_image(nifti_t2, target_resolution, use_voxels=False, interp_type=3)
    nifti_flair = ants.resample_image(nifti_flair, target_resolution, use_voxels=False, interp_type=3)
    nifti_brain_mask = ants.resample_image_to_target(nifti_brain_mask, nifti_t1, interp_type=1)
    tumor_mask = ants.resample_image_to_target(tumor_mask, nifti_t1, interp_type=1)
    CE_tumor_mask = ants.resample_image_to_target(CE_tumor_mask, nifti_t1, interp_type=1)
    NE_tumor_mask = ants.resample_image_to_target(NE_tumor_mask, nifti_t1, interp_type=1)
    WT_tumor_mask = ants.resample_image_to_target(WT_tumor_mask, nifti_t1, interp_type=1)
    
    nifti_t1 = ants.crop_image(nifti_t1, label_image=nifti_brain_mask, label=1)
    nifti_t1ce = ants.crop_image(nifti_t1ce, label_image=nifti_brain_mask, label=1)
    nifti_t2 = ants.crop_image(nifti_t2, label_image=nifti_brain_mask, label=1)
    nifti_flair = ants.crop_image(nifti_flair, label_image=nifti_brain_mask, label=1)
    tumor_mask = ants.crop_image(tumor_mask, label_image=nifti_brain_mask, label=1)
    CE_tumor_mask = ants.crop_image(CE_tumor_mask, label_image=nifti_brain_mask, label=1)
    NE_tumor_mask = ants.crop_image(NE_tumor_mask, label_image=nifti_brain_mask, label=1)
    WT_tumor_mask = ants.crop_image(WT_tumor_mask, label_image=nifti_brain_mask, label=1)
    nifti_brain_mask = ants.crop_image(nifti_brain_mask, label_image=nifti_brain_mask, label=1)
    
    nifti_t1_shape = nifti_t1.shape
    slice_index = [[(nifti_t1_shape[i] - target_shape[i]) // 2, (nifti_t1_shape[i] + target_shape[i]) // 2] if nifti_t1_shape[i] > target_shape[i] else [0, nifti_t1_shape[i]] for i in range(3)]
    nifti_t1 = ants.crop_indices(nifti_t1, [slice_index[0][0], slice_index[1][0], slice_index[2][0]], [slice_index[0][1], slice_index[1][1], slice_index[2][1]])
    nifti_t1ce = ants.crop_indices(nifti_t1ce, [slice_index[0][0], slice_index[1][0], slice_index[2][0]], [slice_index[0][1], slice_index[1][1], slice_index[2][1]])
    nifti_t2 = ants.crop_indices(nifti_t2, [slice_index[0][0], slice_index[1][0], slice_index[2][0]], [slice_index[0][1], slice_index[1][1], slice_index[2][1]])
    nifti_flair = ants.crop_indices(nifti_flair, [slice_index[0][0], slice_index[1][0], slice_index[2][0]], [slice_index[0][1], slice_index[1][1], slice_index[2][1]])
    nifti_brain_mask = ants.crop_indices(nifti_brain_mask, [slice_index[0][0], slice_index[1][0], slice_index[2][0]], [slice_index[0][1], slice_index[1][1], slice_index[2][1]])
    tumor_mask = ants.crop_indices(tumor_mask, [slice_index[0][0], slice_index[1][0], slice_index[2][0]], [slice_index[0][1], slice_index[1][1], slice_index[2][1]])
    CE_tumor_mask = ants.crop_indices(CE_tumor_mask, [slice_index[0][0], slice_index[1][0], slice_index[2][0]], [slice_index[0][1], slice_index[1][1], slice_index[2][1]])
    NE_tumor_mask = ants.crop_indices(NE_tumor_mask, [slice_index[0][0], slice_index[1][0], slice_index[2][0]], [slice_index[0][1], slice_index[1][1], slice_index[2][1]])
    WT_tumor_mask = ants.crop_indices(WT_tumor_mask, [slice_index[0][0], slice_index[1][0], slice_index[2][0]], [slice_index[0][1], slice_index[1][1], slice_index[2][1]])
    if nifti_t1.shape[0] < target_shape[0] or nifti_t1.shape[1] < target_shape[1] or nifti_t1.shape[2] < target_shape[2]:
        size_x = target_shape[0]
        size_y = target_shape[1]
        size_z = target_shape[2]
        nifti_t1 = ants.pad_image(nifti_t1, shape=(size_x, size_y, size_z), value=0)
        nifti_t1ce = ants.pad_image(nifti_t1ce, shape=(size_x, size_y, size_z), value=0)
        nifti_t2 = ants.pad_image(nifti_t2, shape=(size_x, size_y, size_z), value=0)
        nifti_flair = ants.pad_image(nifti_flair, shape=(size_x, size_y, size_z), value=0)
        nifti_brain_mask = ants.pad_image(nifti_brain_mask, shape=(size_x, size_y, size_z), value=0)
        tumor_mask = ants.pad_image(tumor_mask, shape=(size_x, size_y, size_z), value=0)
        CE_tumor_mask = ants.pad_image(CE_tumor_mask, shape=(size_x, size_y, size_z), value=0)
        NE_tumor_mask = ants.pad_image(NE_tumor_mask, shape=(size_x, size_y, size_z), value=0)
        WT_tumor_mask = ants.pad_image(WT_tumor_mask, shape=(size_x, size_y, size_z), value=0)
        
    nifti_t1_bet = ants.mask_image(nifti_t1, nifti_brain_mask)
    nifti_t1ce_bet = ants.mask_image(nifti_t1ce, nifti_brain_mask)
    nifti_t2_bet = ants.mask_image(nifti_t2, nifti_brain_mask)
    nifti_flair_bet = ants.mask_image(nifti_flair, nifti_brain_mask)
    
    nifti_t1_bet.to_file(os.path.join(target_dir, 't1.nii.gz'))
    nifti_t1ce_bet.to_file(os.path.join(target_dir, 't1ce.nii.gz'))
    nifti_t2_bet.to_file(os.path.join(target_dir, 't2.nii.gz'))
    nifti_flair_bet.to_file(os.path.join(target_dir, 'flair.nii.gz'))
    nifti_brain_mask.to_file(os.path.join(target_dir, 'brain_mask.nii.gz'))
    tumor_mask.to_file(os.path.join(target_dir, 'tumor_mask.nii.gz'))
    CE_tumor_mask.to_file(os.path.join(target_dir, 'CE_tumor_mask.nii.gz'))
    NE_tumor_mask.to_file(os.path.join(target_dir, 'NE_tumor_mask.nii.gz'))
    WT_tumor_mask.to_file(os.path.join(target_dir, 'WT_tumor_mask.nii.gz'))
    
    # remove bet files
    os.system(f'rm {os.path.join(target_dir, "t1_bet.nii.gz")}')
    os.system(f'rm {os.path.join(target_dir, "t1ce_bet.nii.gz")}')
    os.system(f'rm {os.path.join(target_dir, "t2_bet.nii.gz")}')
    os.system(f'rm {os.path.join(target_dir, "flair_bet.nii.gz")}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MRI data preprocessing for GBM survival prediction')
    parser.add_argument('--source_root', type=str, required=True,
                        help='Source directory containing raw MRI data')
    parser.add_argument('--target_root', type=str, required=True,
                        help='Target directory for preprocessed data')
    parser.add_argument('--processes', type=int, default=8,
                        help='Number of processes for parallel processing (default: 8)')
    
    args = parser.parse_args()
    
    source_root = args.source_root
    target_root = args.target_root
    num_processes = args.processes
    
    os.makedirs(target_root, exist_ok=True)
    
    patient_dir_list = glob(os.path.join(source_root, '*'))
    
    # patient_dir_list = patient_dir_list[:1]
    print(f"Processing {len(patient_dir_list)} patients from {source_root} to {target_root}")
    print(f"Using {num_processes} processes")
    print(f"First patient directory: {patient_dir_list[0] if patient_dir_list else 'None'}")
    
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(patient_dir_list)) as pbar:
            for _ in pool.imap_unordered(preprocessing_mri, patient_dir_list):
                pbar.update()