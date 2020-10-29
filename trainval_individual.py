import sys
import os
import importlib
import numpy as np
import torch
import torch.nn as nn
from core.classificationnet import ClassificationNet
from core.utils import build_backbone_info, build_imagedataloaders
from core.utils import save_checkpoint, load_checkpoint, save_val_record
from core.utils import build_optimizers, build_schedulers
from algorithms import train_epoch, test_epoch, ModelCheckpoint


def run_trainval(model, train_type, dataset, max_epoch, device, checkpoint_dir,
                 train_loader, val_loader, optimizers, schedulers, save_opt):

    title_str = '== TRAINVAL {} on {} =='.format(train_type, dataset)
    bound_str = '=' * len(title_str)
    print(bound_str + '\n' + title_str + '\n' + bound_str)
    print('Checkpoint Directory: {}'.format(checkpoint_dir))
    output_dir, inner_chkpt = os.path.split(checkpoint_dir)
    model_checkpoint = ModelCheckpoint(-1, checkpoint_dir, save_opt, max_epoch)

    for epoch_idx in range(1, max_epoch+1):

        train_loss, train_acc = train_epoch(
                model, device, train_loader, optimizers, epoch_idx)

        val_loss, val_acc = test_epoch(
                model, device, val_loader, epoch_idx)

        model_checkpoint(val_acc, epoch_idx, model)

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
    assert(args.train_type in ['baseline', 'finetune'])
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
    #----------------------------------
    param_types = args.param_types
    max_epoch   = args.max_epoch
    optimizer_infos = args.optimizer_infos
    scheduler_infos = args.scheduler_infos

    # ---------------------------------
    # Backbone settings
    # ---------------------------------
    backbone_info = build_backbone_info(args.backbone, 'standard', image_size)

    # ---------------------------------
    # Method settings
    # ---------------------------------
    experiment_dir = 'CHECKPOINTS/Individual/{}/{}/{}'.format(
            args.exp_name, args.backbone, args.dataset)

    if args.pretrain != '':
        assert(args.train_type != 'baseline'), 'Cannot use pretrain in baseline train_type'
        print('Load from the pretrained model!')
        model, _ = load_checkpoint(args.pretrain)

    else:
        assert(args.train_type != 'finetune'), 'Cannot use finetune train_type without pretrain'
        model = ClassificationNet(backbone_info, args.num_classes)

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
    inner_chkpt = args.train_type + args.chkpt_postfix
    checkpoint_dir = os.path.join(experiment_dir, inner_chkpt)

    # Get the optimizers and schedulers
    optimizers = build_optimizers(model.module, param_types, optimizer_infos)
    schedulers = build_schedulers(optimizers, scheduler_infos)

    # Run training and validation loops
    run_trainval(
            model, args.train_type, args.dataset, max_epoch, device, checkpoint_dir,
            train_loader, val_loader, optimizers, schedulers, args.save_opt)
    return


if __name__ == '__main__':
    main()
