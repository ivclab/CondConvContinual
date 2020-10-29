import numpy as np
import torch
import torch.nn as nn
from .layers import CondConv2d
from .register import registered_backbones


class ClassificationNet(nn.Module):
    def __init__(self, backbone_info, num_classes=-1):
        super(ClassificationNet, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

        # Building the model structure
        self.build_backbone(backbone_info)

        # Building the classification head
        if num_classes != -1:
            self.build_classification_head(num_classes)
        return

    def compact_info(self):
        model_info = {'type': self.__class__.__name__, 'args': {}}
        model_info['args']['backbone_info'] = self.backbone.compact_info()
        return model_info

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def build_backbone(self, info):
        self.backbone = registered_backbones[info['type']](**info['args'])
        self.feat_dim = np.prod(self.backbone.feat_dim)
        return

    def build_classification_head(self, num_classes):
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        return

    def extract_backbone_conv_modules(self):
        avalable_conv_modules = (nn.Conv2d, CondConv2d)
        backbone_conv_modules = [('backbone.'+n, m) for n, m in self.backbone.named_modules()
                                 if isinstance(m, avalable_conv_modules)]
        return dict(backbone_conv_modules)

    def parse_structure(self):
        backbone_conv_modules = self.extract_backbone_conv_modules()
        shared_parts = [n+'.weight' for n in backbone_conv_modules.keys()]
        model_state_dict = self.state_dict()
        exclusive_parts = [n for n in model_state_dict.keys() if n not in shared_parts]
        return shared_parts, exclusive_parts
