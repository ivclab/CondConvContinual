from .register import register_module
import math
from torch.optim import lr_scheduler


@register_module('schedulers')
class ConstantLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)
        return

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]


@register_module('schedulers')
class LookUpTableLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, look_up_table, last_epoch=-1):
        self.look_up_table = look_up_table
        super(LookUpTableLR, self).__init__(optimizer, last_epoch)
        return

    def get_lr(self):
        new_lr = next((lr for max_epoch, lr in self.look_up_table if max_epoch > self.last_epoch),
                      self.look_up_table[-1][1])
        return [new_lr for _ in self.base_lrs]


@register_module('schedulers')
class EpochBasedExponentialLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, decay=0.96, last_epoch=-1):
        self.decay = decay
        super(EpochBasedExponentialLR, self).__init__(optimizer, last_epoch)
        return

    def get_lr(self):
        return [base_lr * (self.decay ** self.last_epoch)
                for base_lr in self.base_lrs]


@register_module('schedulers')
class EpochBasedCosineLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_epochs, last_epoch=-1):
        self.max_epochs = max_epochs
        super(EpochBasedCosineLR, self).__init__(optimizer, last_epoch)
        return

    def get_lr(self):
        return [0.5 * base_lr * (1 + math.cos(math.pi * self.last_epoch / self.max_epochs))
                for base_lr in self.base_lrs]
