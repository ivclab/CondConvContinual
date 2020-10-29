import sys
import os
import json
import argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from .datasets import ImageDataset, TransformManager
from .dataloaders import ImageDataLoader
from .task_manager import TaskManager
from .layers import CondConv2d
from .classificationnet import ClassificationNet
from .register import registered_schedulers, registered_optimizers


def convert_standardconv_to_condconv(source_chkpt_dir, target_chkpt_dir, task_idx, dataset):

    assert(task_idx == 1)

    print('Converting the model ... ')
    model, _, = load_checkpoint(source_chkpt_dir)
    model_state_dict = model.state_dict()
    model_info = model.compact_info()
    model_args = model_info['args']
    model_args['backbone_info']['args']['conv_type'] = 'cond'
    model_args['backbone_info']['args']['num_experts'] = 1

    # Build the pickingcondconv model and copy the parameters
    model = ClassificationNet(**model_args)
    num_classes = model_state_dict['classifier.weight'].size(0)
    model.build_classification_head(num_classes)

    module_dict = dict(
        (n, m) for n, m in model.named_modules() if len(m._modules) == 0
    )
    for name, param in model.state_dict().items():

        if name in model_state_dict:
            param.data.copy_(model_state_dict[name].view(param.data.size()))
        else:
            module_name, param_name = name.rsplit('.', 1)
            module = module_dict[module_name]

            assert(isinstance(module, CondConv2d)), ('The difference should be conv layers.')

            if param_name == 'routing_weight':
                tensor = torch.zeros_like(param.data)
            elif param_name == 'routing_bias':
                tensor = torch.ones_like(param.data) * float('inf')
            else:
                raise NotImplementedError('Unexpected param found!')

            param.data.copy_(tensor)

    # Build the task manager
    manager = TaskManager(model=model, init_task_idx=task_idx)

    # Save the manager and the converted model
    manager.save_task_dataset(task_idx, dataset)
    manager.save_task_exclusive_params(model, task_idx)
    save_checkpoint(model=model, manager=manager, chkpt_dir=target_chkpt_dir)
    return


def find_task_dir_by_idx(experiment_dir, task_idx):
    task_dirs = os.listdir(experiment_dir)
    for index, task_dir in enumerate(os.listdir(experiment_dir)):
        task_name, _ = task_dir.split('_', 1)
        if task_name[:4] != 'Task':
            continue
        found_idx = int(task_name[4:])
        if task_idx == found_idx:
            break
        elif index == len(task_dirs) - 1:
            raise ValueError(('Cannot find task_dir for '
                              'task_idx {}'.format(task_idx)))
    return task_dir


def save_checkpoint(model=None, manager=None, chkpt_dir=''):
    assert(chkpt_dir != ''), 'Need to specify checkpoint directory'
    os.makedirs(chkpt_dir, exist_ok=True)

    # Save the model
    if model is not None:
        save_path = os.path.join(chkpt_dir, 'model.pth')
        content = {}
        content['model_info'] = model.compact_info()
        content['model_state_dict'] = model.state_dict()
        torch.save(content, save_path)

    # Save the manager
    if manager is not None:
        save_path = os.path.join(chkpt_dir, 'manager.pth')
        content = {}
        content['task_exclusive_params'] = manager.task_exclusive_params
        content['shared_parts']          = manager.shared_parts
        content['exclusive_parts']       = manager.exclusive_parts
        content['task_weight_masks']     = manager.task_weight_masks
        content['task_datasets']         = manager.task_datasets
        torch.save(content, save_path)
    return


def load_checkpoint(chkpt_dir):
    assert(chkpt_dir != ''), 'Need to specify chkpt_dir'
    model, manager = None, None

    # Build and load the model
    load_path = os.path.join(chkpt_dir, 'model.pth')
    if os.path.exists(load_path):
        content = torch.load(load_path)
        model_info = content['model_info']
        assert(model_info['type'] == 'ClassificationNet')
        model = ClassificationNet(**model_info['args'])

        num_classes = content['model_state_dict']['classifier.weight'].size(0)
        model.build_classification_head(num_classes)
        model.load_state_dict(content['model_state_dict'])

    # Build and load the manager
    load_path = os.path.join(chkpt_dir, 'manager.pth')
    if os.path.exists(load_path):
        content = torch.load(load_path)
        manager = TaskManager(chkpt_content=content)

    return model, manager


def build_imagedataloaders(mode, dataset_name, transform_name, image_size, batch_size, padding,
                           save_opt, workers):

    kwargs = {'num_workers': workers, 'pin_memory': True}
    transform_manager = TransformManager(image_size, name=transform_name, padding=padding)
    train_transform = transform_manager.build_composed_transform(augmentation=True)
    test_transform  = transform_manager.build_composed_transform(augmentation=False)
    splitdir = os.path.join('splits', dataset_name)

    if mode == 'trainval':
        train_dataset = ImageDataset(os.path.join(splitdir, 'train.json'), train_transform,
                                     dataset_name)
        train_loader  = ImageDataLoader(train_dataset, batch_size=batch_size,
                                        shuffle=True, **kwargs)

        meta_file = os.path.join(splitdir, 'val.json')
        if not os.path.exists(meta_file):
            assert(save_opt == 'last'), 'We are about to use test loader. Only compatable with save_last'
            meta_file = os.path.join(splitdir, 'test.json')

        val_dataset = ImageDataset(meta_file, test_transform, dataset_name)
        val_loader  = ImageDataLoader(val_dataset, batch_size=batch_size,
                                      shuffle=False, **kwargs)
        return train_loader, val_loader

    else:
        meta_file = os.path.join(splitdir, 'test.json')
        test_dataset = ImageDataset(meta_file, test_transform, dataset_name)
        test_loader  = ImageDataLoader(test_dataset, batch_size=batch_size,
                                       shuffle=False, **kwargs)
        return test_loader


def build_backbone_info(backbone, conv_type='standard', image_size=84,
                        track_running_stats=True, momentum=-1):
    if backbone == 'Conv4':
        backbone_type = 'ConvNet'
        momentum = 0.1 if momentum == -1 else momentum
        backbone_args = {'num_stages': 4, 'in_planes': 3, 'out_planes': 64,
                         'conv_type': conv_type, 'image_size': image_size,
                         'track_running_stats': track_running_stats, 'momentum': momentum}

    elif backbone == 'ResNet18':
        backbone_type = 'ResNet'
        backbone_args = {'block_type': 'BasicBlock', 'nblocks': [2, 2, 2, 2],
                         'conv_type': conv_type, 'image_size': image_size}

    else:
        raise NotImplementedError()
    return {'type': backbone_type, 'args': backbone_args}


class Optimizers(object):
    def __init__(self):
        self.optimizers = []
        return

    def add(self, optimizer):
        self.optimizers.append(optimizer)
        return

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def __getitem__(self, index):
        return self.optimizers[index]

    def __setitem__(self, index, value):
        self.optimizers[index] = value
        return


class Schedulers(object):
    def __init__(self):
        self.schedulers = []
        return

    def add(self, scheduler):
        self.schedulers.append(scheduler)
        return

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()
        return

    def __getitem__(self, index):
        return self.schedulers[index]

    def __setitem__(self, index, value):
        self.schedulers[index] = value
        return


def build_optimizers(model, param_types, optimizer_infos, manager=None, task_idx=-1,
                     masks=None):

    if len(param_types) > 1:
        assert(param_types[-1] == 'other' and not any([x == 'all' for x in param_types]))
    else:
        assert(param_types[0] == 'all'), 'If only one param type, it must be \'all\'.'

    optimizers = Optimizers()
    param_dict = dict(model.named_parameters())
    selected_param_names = []
    for idx, param_type in enumerate(param_types):
        assert(param_type in ['all', 'routing', 'other'])

        optimizer_type = optimizer_infos[idx]['type']
        optimizer_args = optimizer_infos[idx]['args']

        if param_type == 'routing':
            named_params = [(n, p) for n, p in param_dict.items()
                            if 'routing_weight' in n or 'routing_bias' in n]
        elif param_type == 'other':
            named_params = [(n, p) for n, p in param_dict.items()
                            if n not in selected_param_names]

        else:
            named_params = [(n, p) for n, p in param_dict.items()]

        param_names, params = zip(*named_params)
        selected_param_names.extend(param_names)

        if optimizer_type.startswith('Masked'):
            assert((manager is not None and task_idx != -1) or masks is not None)

            if masks is not None:
                mask_keys = list(masks.keys())
                optimizer = registered_optimizers[optimizer_type](
                        params, masks=masks, param_names=param_names,
                        shared_parts=mask_keys, **optimizer_args)
            else:
                task_weight_masks = manager.task_weight_masks
                masks = {n: m.eq(task_idx) for n, m in task_weight_masks.items()}
                optimizer = registered_optimizers[optimizer_type](
                        params, masks=masks, param_names=param_names,
                        shared_parts=manager.shared_parts, **optimizer_args)
        else:
            optimizer = getattr(optim, optimizer_type)(params, **optimizer_args)
        optimizers.add(optimizer)

    return optimizers


def build_schedulers(optimizers, scheduler_infos):

    schedulers = Schedulers()
    for idx, scheduler_info in enumerate(scheduler_infos):

        scheduler_type = scheduler_info['type']
        scheduler_args = scheduler_info['args']
        scheduler = registered_schedulers[scheduler_type](
                optimizers[idx], **scheduler_args)

        schedulers.add(scheduler)

    return schedulers


def save_val_record(output_dir, inner_chkpt, best_val_acc, best_epoch):

    val_record_path = os.path.join(output_dir, 'val_record.json')

    # Read the record if exists
    if not os.path.exists(val_record_path):
        val_record = {}
    else:
        with open(val_record_path, 'r') as f:
            val_record = json.load(f)

    val_record[inner_chkpt] = [best_val_acc, best_epoch]

    # Save the updated record
    with open(val_record_path, 'w') as f:
        json.dump(val_record, f)
    return
