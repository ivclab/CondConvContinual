import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import CondConv2d
from .register import register_module


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, num_experts=1, pool=True, padding=1,
                 momentum=0.1, track_running_stats=True, conv_type='standard'):

        super(ConvBlock, self).__init__()
        assert(conv_type in ['standard', 'cond'])
        self.conv_type = conv_type

        self.layers = nn.Sequential()

        if conv_type == 'cond':
            self.layers.add_module('Conv', CondConv2d(
                in_planes, out_planes, kernel_size=3, stride=1, padding=padding,
                bias=False, num_experts=num_experts))
        else:
            self.layers.add_module('Conv', nn.Conv2d(
                in_planes, out_planes, kernel_size=3, stride=1, padding=padding,
                bias=False))

        self.layers.add_module('BatchNorm', nn.BatchNorm2d(
            out_planes, momentum=momentum, track_running_stats=track_running_stats))
        self.layers.add_module('ReLU', nn.ReLU(inplace=True))

        if pool:
            self.layers.add_module(
                'MaxPool', nn.MaxPool2d(kernel_size=2, stride=2))
        return

    def forward(self, x):
        x = self.layers(x)
        return x


@register_module('backbones')
class ConvNet(nn.Module):
    def __init__(self, in_planes, out_planes, num_stages, num_experts=1,
                 conv_type='standard', image_size=84, momentum=0.1, track_running_stats=True):

        super(ConvNet, self).__init__()
        self.num_stages = num_stages
        self.num_experts = num_experts
        self.in_planes  = in_planes
        self.conv_type  = conv_type
        self.image_size = image_size
        self.track_running_stats = track_running_stats
        self.momentum = momentum

        if type(out_planes) == int:
            self.out_planes = [out_planes for i in range(num_stages)]
        else:
            self.out_planes = out_planes

        assert(type(self.out_planes)==list and len(self.out_planes)==self.num_stages)

        num_planes = [self.in_planes,] + self.out_planes
        conv_blocks = []

        for i in range(self.num_stages):
            conv_block = ConvBlock(num_planes[i], num_planes[i+1], num_experts,
                                   momentum=momentum, track_running_stats=track_running_stats,
                                   conv_type=conv_type)
            conv_blocks.append(conv_block)

        self.conv_blocks = nn.Sequential(*conv_blocks)

        if image_size == 32:
            sdim = (2, 2)  # Only for 32 x 32 input size to Conv4
        else:
            raise NotImplementedError()

        cdim = (self.out_planes[-1],)
        self.feat_dim = cdim + sdim
        return

    def forward(self, x):
        x = self.conv_blocks(x)
        return x

    def compact_info(self):
        backbone_info = {'type': self.__class__.__name__, 'args': {}}
        backbone_info['args']['num_stages']  = self.num_stages
        backbone_info['args']['num_experts'] = self.num_experts
        backbone_info['args']['in_planes']   = self.in_planes
        backbone_info['args']['out_planes']  = self.out_planes
        backbone_info['args']['conv_type']   = self.conv_type
        backbone_info['args']['image_size']  = self.image_size
        backbone_info['args']['track_running_stats'] = self.track_running_stats
        backbone_info['args']['momentum']    = self.momentum
        return backbone_info


def conv3x3(in_planes, out_planes, stride=1, num_experts=1, conv_type='standard'):

    assert(conv_type in ['standard', 'cond'])

    if conv_type == 'standard':
        return nn.Conv2d(
                in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    else:
        return CondConv2d(
                in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
                num_experts=num_experts)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=None,
                 num_experts=1, conv_type='standard'):
        super(BasicBlock, self).__init__()
        self.layers = nn.Sequential(
            conv3x3(in_planes, planes, stride, num_experts=num_experts, conv_type=conv_type),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            conv3x3(planes, planes, num_experts=num_experts, conv_type=conv_type),
            nn.BatchNorm2d(planes),
        )
        self.shortcut = shortcut
        return

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y


@register_module('backbones')
class ResNet(nn.Module):
    def __init__(self, block_type, nblocks, num_experts=1, conv_type='standard',
                 image_size=32):
        super(ResNet, self).__init__()

        assert(block_type in ['BasicBlock', 'Bottleneck'])
        self.block_type = block_type
        self.nblocks = nblocks
        self.num_experts = num_experts
        self.conv_type = conv_type
        self.image_size = image_size

        if block_type == 'BasicBlock':
            block = BasicBlock
        else:
            raise NotImplementedError()

        self.in_planes = 64
        self.pre_layers = nn.Sequential(
            conv3x3(3,64, num_experts=num_experts, conv_type=conv_type),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.layer1 = self._make_layer(block, 64, nblocks[0])
        self.layer2 = self._make_layer(block, 128, nblocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, nblocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, nblocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)

        if image_size == 32:
            sdim = (1, 1)  # Only for 32 x 32 input size to ResNet18
        else:
            raise NotImplementedError()

        cdim = (512,)
        self.feat_dim = cdim + sdim
        return

    def _make_layer(self, block, planes, nblocks, stride=1):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:

            if self.conv_type == 'standard':
                conv_layer = nn.Conv2d(
                        self.in_planes, planes * block.expansion, kernel_size=1,
                        stride=stride, bias=False)
            else:
                conv_layer = CondConv2d(
                        self.in_planes, planes * block.expansion, kernel_size=1,
                        stride=stride, bias=False, num_experts=self.num_experts)

            shortcut = nn.Sequential(
                conv_layer,
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut,
                            num_experts=self.num_experts, conv_type=self.conv_type))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, num_experts=self.num_experts,
                                conv_type=self.conv_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x

    def compact_info(self):
        backbone_info = {'type': self.__class__.__name__, 'args': {}}
        backbone_info['args']['block_type']  = self.block_type
        backbone_info['args']['nblocks']     = self.nblocks
        backbone_info['args']['num_experts'] = self.num_experts
        backbone_info['args']['conv_type']   = self.conv_type
        backbone_info['args']['image_size']  = self.image_size
        return backbone_info
