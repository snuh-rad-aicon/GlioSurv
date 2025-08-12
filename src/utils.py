import os
import sys

import random
import builtins
import warnings
import numpy as np

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import argparse
from omegaconf import OmegaConf
from collections import deque

def set_seed(seed=None):
    if seed is not None:
        # Set environment variables for deterministic behavior
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Set basic random seeds
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 멀티 GPU 용
        np.random.seed(seed)
        
        # Set deterministic settings
        cudnn.deterministic = True
        cudnn.benchmark = False

        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting '
                      'and PyTorch deterministic algorithms, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints. '
                      'For complete reproducibility, set PYTHONHASHSEED environment variable.')
    else:
        cudnn.benchmark = True

def dist_setup(ngpus_per_node, args):
    torch.multiprocessing.set_start_method('fork', force=True)
    # suppress printing if not master
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        dist.barrier()

def get_conf(test=False):
    # First of all, parse config file
    conf_file = sys.argv[1]
    assert os.path.exists(conf_file), f"Config file {conf_file} does not exist!"
    conf = OmegaConf.load(conf_file)

    parser = argparse.ArgumentParser(description='GlioSurv')
    parser.add_argument('conf_file', type=str, help='path to config file')

    for key in conf:
        if conf[key] is None:
            parser.add_argument(f'--{key}', default=None)
        else:
            if key == 'gpu':
                parser.add_argument('--gpu', type=int, default=conf[key])
            elif key == 'multiprocessing_distributed':
                parser.add_argument('--multiprocessing_distributed', type=bool, default=conf[key])
            else:
                parser.add_argument(f'--{key}', type=type(conf[key]), default=conf[key])
                
    args = parser.parse_args()
    args.test = test
    if args.gpu:
        args.gpu = int(args.gpu)

    args.run_name = args.model_name
    args.output_dir = os.path.join('output', args.run_name)
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpts')

    return args

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
        # print(f'count: {self.count} | total: {self.total}')

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def get_vit_layer_id(var_name, num_max_layer, prefix=''):
    if var_name in (prefix + ".cls_token", prefix + ".mask_token", prefix + ".pos_embed"):
        return 0
    elif var_name.startswith(prefix + ".patch_embed") or var_name.startswith(prefix + ".age_embed") or var_name.startswith(prefix + ".gender_embed"):
        return 0
    elif var_name.startswith(prefix + ".rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith(prefix + ".blocks"):
        names = var_name.split('.')
        anchor_ind = names.index('blocks') # 'blocks' is an anchor
        block_id = int(names[anchor_ind + 1])
        offset = 1
        return block_id + offset
    else:
        return num_max_layer - 1

class LayerDecayValueAssigner(object):
    def __init__(self, layer_decay, num_layers):
        self.values = {}
        if 'mm_encoder' in num_layers:
            num_layers_encoder = num_layers['encoder']
            num_layers_clinical_encoder = num_layers['clinical_encoder']
            num_layers_mm_encoder = num_layers['mm_encoder']
            self.values['encoder'] = list(layer_decay ** (num_layers_encoder + num_layers_mm_encoder + 1 - i) for i in range(num_layers_encoder + 2))
            self.values['clinical_encoder'] = list(layer_decay ** (num_layers_clinical_encoder + num_layers_mm_encoder + 1 - i) for i in range(num_layers_clinical_encoder + 2))
            self.values['mm_encoder'] = list(layer_decay ** (num_layers_encoder + num_layers_mm_encoder + 1 - i) for i in range(num_layers_encoder + 1, num_layers_encoder + num_layers_mm_encoder + 2))
            
            self.values['concept_decoder'] = list(layer_decay ** (num_layers_clinical_encoder + num_layers_mm_encoder + 1 - i) for i in range(num_layers_clinical_encoder + 2))
            self.values['decoder'] = list(layer_decay ** (1 + 1 - i) for i in range(1 + 2))
        else:
            num_layers_encoder = num_layers['encoder']
            self.values['encoder'] = list(layer_decay ** (num_layers_encoder + 1 - i) for i in range(num_layers_encoder + 2))
            self.values['decoder'] = list(layer_decay ** (1 + 1 - i) for i in range(1 + 2))
            
        print(f"LayerDecayValueAssigner: {self.values}")

    def get_scale(self, layer_id, prefix=''):
        if layer_id is not None:
            return self.values[prefix][layer_id]
        else:
            return 1

    def get_layer_id(self, var_name, prefix=''):
        return get_vit_layer_id(var_name, len(self.values[prefix]), prefix)


@torch.no_grad()
def concat_all_gather(tensor, distributed=True):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if distributed:
        dist.barrier()
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(dist.get_world_size())]
        # print(f"World size: {dist.get_world_size()}")
        dist.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output
    else:
        return tensor
