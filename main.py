import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

import torch
import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)
import wandb

# import sys
# sys.path.append('src/')

from src.utils import set_seed, dist_setup, get_conf
import src.trainers as trainers


def main():
    args = get_conf(False)
    
    set_seed(args.seed)

    if not args.multiprocessing_distributed and args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    if args.multiprocessing_distributed:
        print(f"multiprocessing_distributed: {args.multiprocessing_distributed}")
        ngpus_per_node = torch.cuda.device_count()
        args.ngpus_per_node = ngpus_per_node
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, 
                nprocs=ngpus_per_node, 
                args=(args,))
    else:
        print("single process")
        main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    ngpus_per_node = args.ngpus_per_node
    dist_setup(ngpus_per_node, args)

    trainer_class = getattr(trainers, f'{args.trainer_name}', None)
    trainer = trainer_class(args)
    
    if args.rank == 0:
        args.wandb_id = wandb.util.generate_id()
        run = wandb.init(project=f"{args.proj_name}",
                        name=args.run_name,
                        config=vars(args),
                        id=args.wandb_id,
                        resume='allow',
                        dir=args.output_dir)
    
    trainer.build_model()
    trainer.build_optimizer()
    if args.resume:
        trainer.resume()
    trainer.build_dataloader()

    trainer.run()

    if args.rank == 0:
        run.finish()


if __name__ == '__main__':
    main()
