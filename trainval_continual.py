import sys
import os
import json
from copy import deepcopy
import importlib
import numpy as np
import torch
import torch.nn as nn
from core.utils import convert_standardconv_to_condconv, find_task_dir_by_idx
from core.utils import load_checkpoint, save_checkpoint, build_backbone_info
from core.utils import build_optimizers, build_schedulers, build_imagedataloaders
from algorithms import train_epoch, test_epoch
from algorithms import ModelCheckpoint


def run_trainval(model, manager, task_idx, dataset, max_epoch, device, checkpoint_dir,
                 train_loader, val_loader, optimizers, schedulers, save_opt):

    title_str = '== TRAINVAL FINETUNE Task {} on {} =='.format(task_idx, dataset)
    bound_str = '=' * len(title_str)
    print(bound_str + '\n' + title_str + '\n' + bound_str)
    print('Checkpoint Directory: {}'.format(checkpoint_dir))
    output_dir, inner_chkpt = os.path.split(checkpoint_dir)
    manager.save_task_exclusive_params(model.module, task_idx)
    manager.save_task_dataset(task_idx, dataset)
    model_checkpoint = ModelCheckpoint(task_idx, checkpoint_dir, save_opt, max_epoch)

    for epoch_idx in range(1, max_epoch+1):

        train_loss, train_acc = train_epoch(
                model, device, train_loader, optimizers, epoch_idx)

        val_loss, val_acc = test_epoch(
                model, device, val_loader, epoch_idx)

        model_checkpoint(val_acc, epoch_idx, model, manager=manager)

        schedulers.step()
    return


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
    assert(args.save_opt in ['best', 'last'])

    # ---------------------------------
    # Dataset settings
    # ---------------------------------
    image_size = args.image_size
    batch_size = args.batch_size
    padding = args.padding
    transform_name = args.transform_name

    # ---------------------------------
    # Optimizer and Scheduler settings
    # ---------------------------------
    param_types = args.param_types
    max_epoch   = args.max_epoch
    optimizer_infos = args.optimizer_infos
    scheduler_infos = args.scheduler_infos

    # ---------------------------------
    # Backbone settings
    # ---------------------------------
    backbone_info = build_backbone_info(args.backbone, 'cond', image_size)

    # ---------------------------------
    # Method settings
    # ---------------------------------
    experiment_dir = 'CHECKPOINTS/Continual/{}/{}'.format(
            args.exp_name, args.backbone)

    if args.task_idx == 1:

        # Convert the scratch model with standard conv to cond conv
        source_chkpt_dir = 'CHECKPOINTS/Individual/{}/{}/{}/baseline'.format(
                args.exp_name, args.backbone, args.dataset)
        target_chkpt_dir = os.path.join(
                experiment_dir, 'Task{}_{}'.format(args.task_idx, args.dataset), 'finetune')
        convert_standardconv_to_condconv(
                source_chkpt_dir, target_chkpt_dir, args.task_idx, args.dataset)
        return  # No need training after conversion

    else:

        # Load the model from the previous task
        prev_task_dir  = find_task_dir_by_idx(experiment_dir, args.task_idx - 1)
        prev_chkpt_dir = os.path.join(experiment_dir, prev_task_dir, 'finetune')
        model, manager = load_checkpoint(prev_chkpt_dir)
        manager.rebuild_structure_with_expansion(
                model, args.task_idx, num_classes=args.num_classes,
                zero_init_expand=args.zero_init_expand)

    # ---------------------------------
    # Build the parallel model
    # ---------------------------------
    model = nn.DataParallel(model.to(device))

    # ---------------------------------
    # Run trainval or evaluate
    # ---------------------------------
    # Build the train and validation dataloaders
    train_loader, val_loader = build_imagedataloaders(
            'trainval', os.path.join(args.exp_name, args.dataset), transform_name,
            image_size, batch_size, padding, args.save_opt, args.workers)

    # Get the checkpoint directory name
    checkpoint_dir = os.path.join(
            experiment_dir, 'Task{}_{}'.format(args.task_idx, args.dataset), 'finetune')

    # Get the optimizers and schedulers
    optimizers = build_optimizers(
            model.module, param_types, optimizer_infos, manager=manager, task_idx=args.task_idx)
    schedulers = build_schedulers(
            optimizers, scheduler_infos)

    # Run the training validation
    run_trainval(
            model, manager, args.task_idx, args.dataset, max_epoch, device, checkpoint_dir,
            train_loader, val_loader, optimizers, schedulers, args.save_opt)
    return


if __name__ == '__main__':
    main()
