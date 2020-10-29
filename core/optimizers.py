from .register import register_module
import math
import numpy as np
import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required


def is_all_finite(tensor):
    is_inf = ~((tensor + 1) != tensor)
    is_nan = ~(tensor == tensor)
    not_finite = is_inf | is_nan
    return torch.sum(not_finite).item() == 0


def mask_out_gradients(grad, param_name, masks, shared_parts):

    # Mask out gradients to avoid updating weights
    if param_name in shared_parts:
        grad[masks[param_name].eq(0)] = 0.0

    return


@register_module('optimizers')
class MaskedAdam(Optimizer):
    """ Implementation of Adam optimizer with masks keeping track of
        whether weights can be updated or not
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, masks=None,
                 param_names=None, shared_parts=None):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(MaskedAdam, self).__init__(params, defaults)
        self.masks = masks
        self.param_names = param_names
        self.shared_parts = shared_parts
        assert(len(self.param_groups) == 1), 'Compact all parameters together'
        return

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
        return

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for param_index, p in enumerate(group['params']):

                # --------------------------------------
                # Computing Adam update as usual
                # --------------------------------------

                # Computing Adam update as usual
                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # --------------------------------------
                # Mask out weights not allowd to update
                # --------------------------------------
                param_name = self.param_names[param_index]
                mask_out_gradients(exp_avg, param_name, self.masks, self.shared_parts)

                p.data.addcdiv_(-step_size, exp_avg, denom)
        return loss


@register_module('optimizers')
class MaskedSGD(Optimizer):
    """ Implementation of stochastic gradient descent with masks keeping
        track of whether weights can be updated or not
    """
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, masks=None,
                 param_names=None, shared_parts=None):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError('Nesterov momentum requires a momentum and zero dampening')

        super(MaskedSGD, self).__init__(params, defaults)
        self.masks = masks
        self.param_names = param_names
        self.shared_parts = shared_parts
        assert(len(self.param_groups) == 1), 'Compact all parameters together'
        for param_name in param_names:
            if 'mask' in param_name: assert(param_name not in shared_parts)
        return

    def __setstate__(self, state):
        super(MaskedSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
        return

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay= group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for param_index, p in enumerate(group['params']):

                # --------------------------------------
                # Computing SGD update as usual
                # --------------------------------------

                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0 and is_all_finite(p.data):
                    d_p.add_(weight_decay, p.data)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # --------------------------------------
                # Mask out weights not allowd to update
                # --------------------------------------
                param_name = self.param_names[param_index]
                mask_out_gradients(d_p, param_name, self.masks, self.shared_parts)

                p.data.add_(-group['lr'], d_p)
        return loss
