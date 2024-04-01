# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
import torch
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from utils.model_util import create_model_and_diffusion, load_pretrained_mdm
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from model.afford_est import AffordEstimation
from diffusion.gaussian_diffusion import AffordDiffusion

def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data_conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        training_stage=1
    )
    data = get_dataset_loader(data_conf)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data, ModelClass=AffordEstimation, DiffusionClass=AffordDiffusion)
    model.to(dist_util.dev())


    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
