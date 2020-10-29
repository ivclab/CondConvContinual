import sys
import os
import importlib
import json
import numpy as np
import torch
import torch.nn as nn
from core.utils import load_checkpoint
from core.utils import build_imagedataloaders, build_backbone_info
from algorithms import test_epoch


def main(*args, **kwargs):

    # ---------------------------------
    # Loading the config
    # ---------------------------------
    config_module = importlib.import_module('configs.'+sys.argv[1])
    args = config_module.args
    print(args)

    # ---------------------------------
    # General settings
    # ---------------------------------
    device = 'cuda'
    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed(args.rng_seed)
    torch.cuda.manual_seed_all(args.rng_seed)
    np.random.seed(args.rng_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---------------------------------
    # Dataset settings
    # ---------------------------------
    image_size = args.image_size
    batch_size = args.batch_size
    padding = args.padding
    transform_name = args.transform_name

    # ---------------------------------
    # backbone settings
    # ---------------------------------
    backbone_info = build_backbone_info(args.backbone, 'standard', image_size)

    # ---------------------------------
    # Method settings
    # ---------------------------------
    experiment_dir = 'CHECKPOINTS/Individual/{}/{}/{}'.format(
            args.exp_name, args.backbone, args.dataset)
    output_path = 'CHECKPOINTS/Individual/{}/{}/RESULTS.json'.format(
            args.exp_name, args.backbone)

    # ---------------------------------
    # Run evaluation
    # ---------------------------------
    inner_chkpt = args.train_type + args.chkpt_postfix
    checkpoint_dir = os.path.join(experiment_dir, inner_chkpt)
    model, _ = load_checkpoint(checkpoint_dir)

    model = nn.DataParallel(model.to(device))

    test_loader = build_imagedataloaders(
            'evaluate', os.path.join(args.exp_name, args.dataset), transform_name,
            image_size, batch_size, padding, args.save_opt, args.workers)

    val_loss, val_acc = test_epoch(model, device, test_loader, -1)

    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            content = json.load(f)
    else:
        content = {}

    content['{}_{}'.format(args.dataset, inner_chkpt)] = round(val_acc, 2)

    with open(output_path, 'w') as f:
        json.dump(content, f)
    return


if __name__ == '__main__':
    main()
