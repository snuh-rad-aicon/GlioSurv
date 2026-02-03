import os
import time
import math
from tqdm import tqdm
import numpy as np
from functools import partial
import nibabel as nib

import torch

from torchsurv.loss.weibull import neg_log_likelihood
from torchsurv.loss.weibull import log_hazard
from torchsurv.metrics.cindex import ConcordanceIndex

import wandb

# import sys
# sys.path.append('..')
from src import models
from src.utils import get_conf, SmoothedValue, concat_all_gather, LayerDecayValueAssigner
from src.trainers.base_trainer import BaseTrainer
from src.data.mri_transforms import get_classification_train_transforms, get_val_transforms
from src.data.mm_datasets import get_train_loader, get_val_loader
from src.data.mm_transforms import clinical_variable_list

from collections import defaultdict


class GlioSurvTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = args.model_name
        self.clinical_variable_list = clinical_variable_list + ['patient_id']
        self.train_loss_dict = {}
        self.val_loss_dict = {}
        self.val_c_index_dict = {}
        self.train_min_loss = 1e5
        self.val_min_loss = 1e5
        self.val_max_c_index = 0

    def build_model(self):
        if self.model_name != 'Unknown' and self.model is None:
            args = self.args
            print(f"=> creating model {self.model_name}")
            
            self.model = getattr(models, self.model_name)(args=args)
            
            # load pretrained weights of vision encoder
            print(f"=> Start loading pretrained weights from {args.vision_pretrain}")
            checkpoint = torch.load(args.vision_pretrain, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            for key in list(state_dict.keys()):
                if key.startswith('encoder.'):
                    state_dict[key[len('encoder.'):]] = state_dict[key]
                    del state_dict[key]
                if key == 'encoder_pos_embed':
                    if self.model.vision_encoder.return_cls_token:
                        pe = torch.zeros([1, 1, state_dict[key].size(-1)])
                        state_dict['pos_embed'] = torch.cat([pe, state_dict[key]], dim=1)
                    else:
                        state_dict['pos_embed'] = state_dict[key]
                    del state_dict[key]
                if key == 'patch_embed.proj.weight' and \
                    state_dict['patch_embed.proj.weight'].shape != self.model.vision_encoder.patch_embed.proj.weight.shape:
                    del state_dict['patch_embed.proj.weight']
                    del state_dict['patch_embed.proj.bias']
                if key == 'pos_embed' and \
                    state_dict['pos_embed'].shape != self.model.vision_encoder.pos_embed.shape:
                    del state_dict[key]
            msg = self.model.vision_encoder.load_state_dict(state_dict, strict=False)
            print(f"=> Finish loading pretrained weights from {args.vision_pretrain}")
            if len(msg.missing_keys) > 0:
                print(f"Missing keys: {msg.missing_keys}")
            
            self.wrap_model()

    def build_optimizer(self):
        assert(self.model is not None and self.wrapped_model is not None), \
                "Model is not created and wrapped yet. Please create model first."
        print("=> creating optimizer")
        args = self.args
        model = self.model
        
        trainable_params = []
        
        for name, param in model.named_parameters():
            if any(component in name for component in ['vision_encoder']):
                param.requires_grad = False
            else:
                param.requires_grad = True
                trainable_params.append(param)
        
        print(f"=> Total parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"=> Trainable parameters: {sum(p.numel() for p in trainable_params)}")
        
        self.optimizer = torch.optim.AdamW(trainable_params, 
                                            lr=args.lr,
                                            betas=(args.beta1, args.beta2),
                                            weight_decay=args.weight_decay)
        
    def build_dataloader(self):
        if self.dataloader is None:
            print("=> creating dataloader")
            args = self.args
            
            train_transform = get_classification_train_transforms()
            self.dataloader = get_train_loader(args, 
                                                batch_size=self.batch_size,
                                                workers=self.workers,
                                                train_transform=train_transform)
            val_transform = get_val_transforms()
            self.val_dataloader = get_val_loader(args, 
                                                batch_size=self.vis_batch_size,
                                                workers=self.workers,
                                                val_transform=val_transform)

            self.iters_per_epoch = len(self.dataloader)
            self.val_iters_per_epoch = len(self.val_dataloader)

            if args.eval_metric == 'c_index':
                self.cox_cindex = ConcordanceIndex()
            else:
                raise NotImplementedError("Only support c_index for now")
        else:
            raise ValueError(f"Dataloader has been created. Do not create twice.")
        print("=> finish creating dataloader")
        
    def run(self):
        args = self.args
        niters = args.start_epoch * self.iters_per_epoch

        pbar = tqdm(range(args.start_epoch, args.epochs), total=args.epochs, desc="Epochs")
        for epoch in pbar:
            if args.distributed:
                self.dataloader.sampler.set_epoch(epoch)
                self.val_dataloader.sampler.set_epoch(epoch)

            # train for one epoch
            niters, loss = self.epoch_train(epoch, niters)
            
            # evaluate after each epoch training
            if epoch == 0 or (epoch + 1) % args.eval_freq == 0:
                val_meters, val_c_index = self.evaluate(epoch=epoch, niters=niters)
                
            val_loss = val_meters['loss'].global_avg
            
            self.train_loss_dict[epoch] = loss
            self.val_loss_dict[epoch] = val_loss
            self.val_c_index_dict[epoch] = val_c_index
            
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                self.val_max_c_index = val_c_index
                self.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'model_name': args.model_name,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, 
                    filename=f'{args.ckpt_dir}/best_c_index.pth.tar'
                )

                if epoch == 0 or (epoch + 1) % args.save_freq == 0:
                    self.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'model_name': args.model_name,
                            'state_dict': self.model.state_dict(),
                            'optimizer' : self.optimizer.state_dict(),
                        }, 
                        filename=f'{args.ckpt_dir}/checkpoint_{epoch:04d}.pth.tar'
                    )
                    
            pbar.set_description(f'Train Epoch {epoch:03d}/{args.epochs} | Loss {loss:.03f} | Val Loss {val_loss:.03f} | Val C-Index {val_c_index:.03f}')

    def epoch_train(self, epoch, niters):
        args = self.args
        train_loader = self.dataloader
        train_dataset = train_loader.dataset
        model = self.wrapped_model
        optimizer = self.optimizer
        
        model.set_train()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=False)
        for i, batch_data in pbar:
            self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)
            
            image = batch_data['input']
            event = batch_data['death_event']
            duration = batch_data['death_duration_month']
            
            if event.sum().item() == 0:
                event_data = train_dataset.get_event_data()
                random_idx = torch.randint(0, len(event), (1,))
                
                image[random_idx] = event_data['input']
                event[random_idx] = event_data['death_event']
                duration[random_idx] = event_data['death_duration_month']
            
            if args.gpu is not None:
                image = image.as_tensor().to(args.gpu, non_blocking=True)
                clinical_variables = {}
                for k, v in batch_data.items():
                    if k in self.clinical_variable_list:
                        if isinstance(v, torch.Tensor):
                            clinical_variables[k] = v.to(args.gpu, non_blocking=True)
                        else:
                            clinical_variables[k] = v
                event = event.cuda(args.gpu, non_blocking=True)
                duration = duration.cuda(args.gpu, non_blocking=True)

            loss = self.train_class_batch(model, image, clinical_variables, event, duration)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % args.print_freq == 0:
                if 'lr_scale' in optimizer.param_groups[0]:
                    last_layer_lr = optimizer.param_groups[0]['lr'] / optimizer.param_groups[0]['lr_scale']
                else:
                    last_layer_lr = optimizer.param_groups[0]['lr']

                pbar.set_description(f'Train Epoch {epoch:03d}/{args.epochs} | Iter {i:05d}/{self.iters_per_epoch} | Init Lr {args.lr:.03f} | Lr {last_layer_lr:.03f} | Loss {loss.item():.03f}')
                if args.rank == 0:
                    wandb.log(
                        {
                        "lr": last_layer_lr,
                        "Loss": loss.item(),
                        },
                        step=niters,
                    )
                    
            niters += 1
        return niters, loss.item()

    @staticmethod
    def train_class_batch(model, samples, clinical_variables, event, duration):
        log_params = model(samples, clinical_variables)
        loss = neg_log_likelihood(log_params, event, duration, reduction="mean")
        return loss
    
    @torch.no_grad()
    def evaluate(self, epoch, niters):
        args = self.args
        model = self.wrapped_model
        val_loader = self.val_dataloader
        meters = defaultdict(SmoothedValue)

        model.set_eval()

        log_params_list = []
        event_list = []
        time_list = []

        pbar = tqdm(enumerate(val_loader), total=len(val_loader), position=0, leave=False)
        for i, batch_data in pbar:
            image = batch_data['input']
            event = batch_data['death_event']
            duration = batch_data['death_duration_month']

            if args.gpu is not None:
                image = image.as_tensor().to(args.gpu, non_blocking=True)
                clinical_variables = {}
                for k, v in batch_data.items():
                    if k in self.clinical_variable_list:
                        if isinstance(v, torch.Tensor):
                            clinical_variables[k] = v.to(args.gpu, non_blocking=True)
                        else:
                            clinical_variables[k] = v
                event = event.to(args.gpu, non_blocking=True)
                duration = duration.to(args.gpu, non_blocking=True)
                
            output = model(image, clinical_variables)
            loss = neg_log_likelihood(output, event, duration, reduction="mean")
            
            log_params_list.append(output)
            event_list.append(event)
            time_list.append(duration)

            batch_size = image.size(0)
            meters['loss'].update(value=loss.item(), n=batch_size)

        if args.distributed:
            for k, v in meters.items():
                v.synchronize_between_processes()
        
        log_params_list = concat_all_gather(torch.cat(log_params_list, dim=0), args.distributed).cpu()
        event_list = concat_all_gather(torch.cat(event_list, dim=0), args.distributed).cpu()
        time_list = concat_all_gather(torch.cat(time_list, dim=0), args.distributed).cpu()
        log_hz_list = log_hazard(log_params_list, time_list)

        c_index = self.cox_cindex(log_hz_list, event_list, time_list).item()

        pbar.set_description(f'Val Epoch {epoch:03d}/{args.epochs} | Loss {meters["loss"].global_avg:.03f} | C-Index {c_index:.03f}')
        if args.rank == 0:
            wandb.log(
                {
                 "Val Loss": meters['loss'].global_avg,
                 "C-Index": c_index,
                },
                step=niters,
            )
            
        return meters, c_index
    