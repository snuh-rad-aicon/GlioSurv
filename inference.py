import os
import numpy as np
import torch

from tqdm import tqdm
from functools import partial

from torchsurv.loss.weibull import cumulative_hazard

from src import models

from src.data.mri_transforms import get_val_transforms
from src.data.mm_datasets import get_test_loader
from src.data.mm_transforms import clinical_variable_list

from src.pytorch_grad_cam_3d.eigen_cam import EigenCAM
from src.pytorch_grad_cam_3d.utils.model_targets import SurvivalOutputTarget

from src.utils import set_seed, get_conf


def create_reshape_transform(model):
    """
    Returns a reshape_transform function for the given model.
    This function is used for attention map visualization.
    """
    def reshape_transform(tensor, height=14, width=14, depth=14):
        # Remove class token if present (ViT-style)
        odd = tensor.size(1) % 2
        if odd:
            result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, depth, tensor.size(2))
        else:
            result = tensor.reshape(tensor.size(0), height, width, depth, tensor.size(2))
        # Change to (batch, channels, H, W, D)
        result = result.permute(0, 4, 1, 2, 3)
        return result

    input_size = model.input_size
    patch_size = model.patch_size
    patch_grid = (
        input_size[0] // patch_size[0],
        input_size[1] // patch_size[1],
        input_size[2] // patch_size[2]
    )
    return partial(
        reshape_transform,
        height=patch_grid[0],
        width=patch_grid[1],
        depth=patch_grid[2]
    )

def main():
    args = get_conf(True)
    set_seed(args.seed)
    
    test_transform = get_val_transforms()
    test_dataloader = get_test_loader(args, batch_size=args.vis_batch_size, workers=args.workers, test_transform=test_transform)
    
    checkpoint = torch.load(args.pretrain, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    model = getattr(models, args.model_name)(args=args)
    model.load_state_dict(state_dict, strict=False)
    model.to(args.gpu)
    model.eval()
    
    target_layers = [model.vision_encoder.blocks[-1].norm1]
    reshape_transform = create_reshape_transform(model)
    cam = EigenCAM(model=model,
                   target_layers=target_layers,
                   reshape_transform=reshape_transform)
    eigen_smooth = True
    
    result_dict = {}
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), position=0, leave=False)
    for i, batch_data in pbar:
        image = batch_data['input']
        event = batch_data['death_event']
        duration = batch_data['death_duration_month']
        patient_id = batch_data['patient_id']
        
        if args.gpu is not None:
            image = image.as_tensor().to(args.gpu, non_blocking=True)
            clinical_variables = {}
            for k, v in batch_data.items():
                if k in clinical_variable_list:
                    if isinstance(v, torch.Tensor):
                        clinical_variables[k] = v.to(args.gpu, non_blocking=True)
                    else:
                        clinical_variables[k] = v

            event = event.to(args.gpu, non_blocking=True)
            duration = duration.to(args.gpu, non_blocking=True)
            
        targets = [SurvivalOutputTarget(duration, event)]
        
        vision_cam = cam(input_tensor=image,
                    condition_tensor=clinical_variables,
                    targets=targets,
                    eigen_smooth=eigen_smooth)

        output = model(image, clinical_variables, save_attn=True)
        loss = cumulative_hazard(output, duration, all_times=False).sum()
        loss.backward(retain_graph=True)
        
        grad1 = model.decoder.blocks[0].cross_attn.attn.grad
        grad2 = model.decoder.blocks[1].cross_attn.attn.grad
        modality_cam = torch.stack([grad1, grad2], dim=1)
        
        for i, (p_id, out, vcam, mcam, ent, dur) in enumerate(zip(patient_id, output, vision_cam, modality_cam, event, duration)):
            save_dir = f'./results/'
            os.makedirs(save_dir, exist_ok=True)
            save_dict = {
                'output': out.cpu().detach().numpy(),
                'vision_cam': vcam,
                'modality_cam': mcam.cpu().detach().numpy(),
                'event': ent.cpu().detach().numpy(),
                'duration': dur.cpu().detach().numpy()
            }
            np.save(f'{save_dir}/{p_id}.npy', save_dict, allow_pickle=True)
            

if __name__ == '__main__':
    main()
