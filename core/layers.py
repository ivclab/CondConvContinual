import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


DEFAULT_THRESHOLD = 5e-3


class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    @staticmethod
    def forward(ctx, inputs):
        outputs = inputs.clone()
        outputs[inputs.le(DEFAULT_THRESHOLD)] = 0
        outputs[inputs.gt(DEFAULT_THRESHOLD)] = 1
        return outputs

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out


class CondConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, num_experts=4,
                 bias=True):

        super(CondConv2d, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.num_experts = num_experts

        self.weight_shape = (self.out_channels, self.in_channels//self.groups) + self.kernel_size
        num_params_in_weight = np.prod(self.weight_shape)
        self.weight = Parameter(torch.Tensor(self.num_experts, num_params_in_weight), requires_grad=True)
        self.routing_weight = Parameter(torch.Tensor(num_experts, in_channels), requires_grad=True)
        self.routing_bias = Parameter(torch.Tensor(num_experts), requires_grad=True)

        if bias:
            self.bias_shape = (self.out_channels,)
            self.bias = Parameter(torch.Tensor(self.num_experts, self.out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        return

    def reset_parameters(self):
        # Initialize expert weight and bias
        for index in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight[index].view(self.weight_shape), a=math.sqrt(5))
        if self.bias is not None:
            fan_in = np.prod(self.weight_shape[1:])
            bound = 1 / math.sqrt(fan_in)
            for index in range(self.num_experts):
                nn.init.uniform_(self.bias[index].view(self.bias_shape), a=-bound, b=bound)

        # Initialize routing weight and bias
        nn.init.kaiming_uniform_(self.routing_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.routing_weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.routing_bias, -bound, bound)
        return

    def forward(self, x):
        pooled_x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        routing = torch.sigmoid(F.linear(pooled_x, self.routing_weight, self.routing_bias))

        B, C, H, W = x.shape
        weight = torch.matmul(routing, self.weight)
        attentioned_weight_shape = (B*self.out_channels, self.in_channels//self.groups) + self.kernel_size
        attentioned_weight = weight.view(attentioned_weight_shape)

        attentioned_bias = None
        if self.bias is not None:
            bias = torch.matmul(routing, self.bias)
            attentioned_bias = bias.view(B * self.out_channels)

        x = x.view(1, B*C, H, W)
        out = F.conv2d(x, attentioned_weight, attentioned_bias, stride=self.stride,
                padding=self.padding, dilation=self.dilation, groups=self.groups*B)
        out = out.permute([1, 0, 2, 3]).view(B, self.out_channels, out.shape[-2], out.shape[-1])
        return out

    def __repr__(self):
        s = ('{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, num_experts={num_experts}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
