import os
import math

import torch
import torch.nn as nn

class BaseTrainer(object):
    r"""
    Base class for all the trainers
    """
    def __init__(self, args):
        self.args = args
        self.model_name = 'Unknown'
        self.model = None
        self.wrapped_model = None
        self.optimizer = None
        self.dataloader = None
        # The following attributes will be modiflied adaptively
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.init_lr() # lr rate
        # create checkpoint directory
        if args.rank == 0:
            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)

    def init_lr(self):
        args = self.args
        # infer learning rate before changing batch size
        self.lr = args.lr

    def wrap_model(self):
        """
        1. Distribute model or not
        2. Rewriting batch size and workers
        """
        args = self.args
        model = self.model
        assert model is not None, "Please build model before wrapping model"
        
        self.batch_size = args.batch_size
        self.vis_batch_size = args.vis_batch_size
        if args.distributed:
            ngpus_per_node = args.ngpus_per_node
            # Apply SyncBN
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                self.batch_size = args.batch_size // ngpus_per_node
                self.vis_batch_size = args.vis_batch_size
                self.workers = (args.workers + ngpus_per_node - 1) // ngpus_per_node
                print("=> Finish adapting batch size and workers according to gpu number")
                model = nn.parallel.DistributedDataParallel(model, 
                                                            device_ids=[args.gpu],
                                                            find_unused_parameters=True)
            else:
                model.cuda()
                model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            # AllGather/rank implementation in this code only supports DistributedDataParallel
            raise NotImplementedError("Must Specify GPU or use DistributeDataParallel.")
        
        self.wrapped_model = model

    def get_parameter_groups(self, get_layer_id=None, get_layer_scale=None, verbose=False):
        args = self.args
        weight_decay = args.weight_decay
        model = self.model
        
        if hasattr(model, 'no_weight_decay'):
            skip_list = model.no_weight_decay()
        else:
            skip_list = {}

        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            
            prefix = name.split('.')[0]
            
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
                # print(f"Adding weight decay to {name}")
                
            if get_layer_id is not None:
                layer_id = get_layer_id(name, prefix=prefix)
                group_name = "%s_layer_%d_%s" % (prefix, layer_id, group_name)
            else:
                layer_id = None

            if group_name not in parameter_group_names:
                if get_layer_scale is not None:
                    scale = get_layer_scale(layer_id, prefix=prefix)
                else:
                    scale = 1.

                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
            
            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
        if verbose:
            import json
            print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
        else:
            print("Param groups information is omitted...")

        return list(parameter_group_vars.values())

    def resume(self):
        args = self.args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']
            self.model.load_state_dict(state_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)

    def adjust_learning_rate(self, epoch, args):
        """Base schedule: CosineDecay with warm-up."""
        init_lr = self.lr
        if epoch < args.warmup_epochs:
            cur_lr = init_lr * epoch / args.warmup_epochs
        else:
            cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            if 'lr_scale' in param_group:
                param_group['lr'] = cur_lr * param_group['lr_scale']
            else:
                param_group['lr'] = cur_lr
    
    def adjust_posemb_rate(self, epoch, args):
        """Base schedule: Cosine Increase"""
        pe_rate = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        return pe_rate