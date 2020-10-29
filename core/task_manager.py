import numpy as np
import torch
from .layers import CondConv2d


def copy_tensor(prev_tensor, curr_tensor):
    if curr_tensor.dim() == 4:
        dim1, dim2, dim3, dim4 = prev_tensor.size()
        curr_tensor[:dim1, :dim2, :dim3, :dim4].copy_(prev_tensor)
    elif curr_tensor.dim() == 2:
        dim1, dim2 = prev_tensor.size()
        curr_tensor[:dim1, :dim2].copy_(prev_tensor)
    elif curr_tensor.dim() == 1:
        dim1 = prev_tensor.size(0)
        curr_tensor[:dim1].copy_(prev_tensor)
    elif curr_tensor.dim() == 0:
        curr_tensor.copy_(prev_tensor)
    else:
        raise NotImplementedError('Unexpected tensor dimension!')
    return


class TaskManager(object):
    def __init__(self, model=None, init_task_idx=1, chkpt_content=None):
        assert(model is not None or chkpt_content is not None), 'Construct task manager use model or chkpt'

        if chkpt_content is not None:
            # Load from the checkpoint
            self.shared_parts          = chkpt_content['shared_parts']
            self.exclusive_parts       = chkpt_content['exclusive_parts']
            self.task_exclusive_params = chkpt_content['task_exclusive_params']
            self.task_datasets         = chkpt_content['task_datasets']
            self.task_weight_masks     = chkpt_content['task_weight_masks']

        else:
            # Build from the given model
            assert(init_task_idx == 1), 'Only build the model for the first task'
            self.task_exclusive_params = {}
            self.task_datasets = {}
            self.shared_parts, self.exclusive_parts = model.parse_structure()
            self.task_weight_masks = {}
            for name, param in model.state_dict().items():
                if name in self.shared_parts:
                    self.task_weight_masks[name] = torch.ones_like(param) * init_task_idx
                else:
                    assert(name in self.exclusive_parts), 'Unexpected param \''+name+'\''
        return

    def save_task_dataset(self, task_idx, dataset_name):
        self.task_datasets[task_idx] = dataset_name
        return

    def save_task_exclusive_params(self, model, task_idx):
        param_dict = {}
        for name, param in model.state_dict().items():
            if name in self.exclusive_parts:
                param_dict[name] = param.data.clone()
        self.task_exclusive_params[task_idx] = param_dict
        return

    def load_task_exclusive_params(self, model, task_idx):
        param_dict = self.task_exclusive_params[task_idx]

        num_classes = param_dict['classifier.weight'].size(0)
        model.build_classification_head(num_classes)

        for name, param in model.state_dict().items():
            if name in self.exclusive_parts:
                param.data.copy_(param_dict[name])
        return

    def rebuild_structure_with_expansion(self, model, task_idx, num_classes=-1,
                                         zero_init_expand=True):

        # Extract backbone info from the model
        backbone_info = model.backbone.compact_info()
        original_num_experts = backbone_info['args']['num_experts']

        # Expand expert columns if needed
        print('Expand the structure!')
        backbone_info['args']['num_experts'] += 1

        # Rebuild the backbone
        prev_state_dict = model.state_dict()
        model.build_backbone(backbone_info)

        # Rebuild the task head
        assert(num_classes != -1)
        model.build_classification_head(num_classes)

        module_dict = dict(
            (n, m) for n, m in model.named_modules() if len(m._modules) == 0
        )
        curr_state_dict = model.state_dict()

        # Copy shared params, and note that the curr exclusive params are reinitialized
        task_weight_masks = {}
        for param_name in self.shared_parts:
            module_name = param_name.rsplit('.', 1)[0]
            module = module_dict[module_name]
            assert(isinstance(module, CondConv2d)), 'Only allow shared CondConv2d.'

            prev_param = prev_state_dict[param_name]
            curr_param = curr_state_dict[param_name]
            assert(prev_param.size(0) == curr_param.size(0) - 1)
            if zero_init_expand:
                curr_param.fill_(0.0)     # HERE: Initialize the expanded params as zero
            copy_tensor(prev_param, curr_param)

            prev_mask = self.task_weight_masks[param_name]
            curr_mask = torch.zeros_like(curr_param)
            assert(prev_mask.size(0) == curr_mask.size(0) - 1)
            copy_tensor(prev_mask, curr_mask)
            curr_mask[-1] = task_idx      # HERE: Acquire params at the expanded column
            task_weight_masks[param_name] = curr_mask

        # Update task exclusive params
        for param_dict in self.task_exclusive_params.values():
            for param_name in param_dict.keys():
                assert(param_name in self.exclusive_parts)
                prev_param = param_dict[param_name]
                module_name, last_name = param_name.rsplit('.', 1)
                module = module_dict[module_name]

                if isinstance(module, CondConv2d):
                    if last_name == 'routing_bias':
                        curr_param = torch.ones_like(curr_state_dict[param_name]) * float('-inf')
                    else:
                        curr_param = torch.zeros_like(curr_state_dict[param_name])
                    assert(prev_param.size(0) == curr_param.size(0) - 1)
                else:
                    curr_param = torch.zeros_like(curr_state_dict[param_name])

                copy_tensor(prev_param, curr_param)
                param_dict[param_name] = curr_param

        # Update the indication masks
        self.task_weight_masks = task_weight_masks
        return
