import sys
import sys
import os
from tqdm import tqdm
from core.utils import save_checkpoint, save_val_record


class ModelCheckpoint(object):
    def __init__(self, task_idx, chkpt_dir, save_opt, max_epoch):
        assert(save_opt in ['best', 'last']), 'Save option must be \'best\' or \'last\'.'
        self.best_val_acc = -1
        self.best_epoch   = -1
        self.task_idx  = task_idx
        self.chkpt_dir = chkpt_dir
        self.save_opt  = save_opt
        self.max_epoch = max_epoch
        return

    def __call__(self, val_acc, epoch_idx, model, manager=None):

        time_to_save_best = (self.save_opt == 'best' and val_acc > self.best_val_acc)
        time_to_save_last = (self.save_opt == 'last' and epoch_idx == self.max_epoch)

        if time_to_save_best or time_to_save_last:

            print('Save the checkpoint!')
            self.best_val_acc = val_acc
            self.best_epoch = epoch_idx

            if manager is not None:
                manager.save_task_exclusive_params(model.module, self.task_idx)

            save_checkpoint(model=model.module, manager=manager,
                            chkpt_dir=self.chkpt_dir)
        return

    def reset(self):
        self.best_val_acc = -1
        self.best_epoch   = -1
        return


class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()
        return

    def update(self, val, count=1):
        self.sum += val * count
        self.count += count
        return

    def reset(self):
        self.sum = 0.0
        self.count = 0.0
        return

    @property
    def avg(self):
        if self.count == 0:
            return 0.0
        else:
            return self.sum / self.count


def classification_accuracy(output, labels):
    preds = output.max(1, keepdim=True)[1]
    return preds.eq(labels.view_as(preds)).cpu().float().mean().item() * 100.


def train_epoch(model, device, train_loader, optimizers, epoch_idx):

    # Building training data iterator
    model.train()
    train_loss = AverageMeter('train_loss')
    train_acc  = AverageMeter('train_acc')
    train_iter = train_loader(epoch_idx-1)
    num_iters  = len(train_loader)

    # Training epoch
    with tqdm(total=num_iters,
              desc='TRAIN Ep. #{}'.format(epoch_idx),
              disable=False,
              dynamic_ncols=True,
              ascii=True) as t:

        for batch_idx, batch_data in enumerate(train_iter):

            optimizers.zero_grad()

            images, labels = batch_data
            images = images.to(device)
            labels = labels.to(device)
            num_samples = labels.size(0)

            output = model(images)
            loss = model.module.loss_fn(output, labels)

            loss.backward()
            optimizers.step()

            train_loss.update(loss.item(), num_samples)
            cls_acc = classification_accuracy(output, labels)
            train_acc.update(cls_acc, num_samples)

            tqdm_postfix = {'train_loss': round(train_loss.avg, 4),
                            'train_acc':  round(train_acc.avg, 4)}

            t.set_postfix(tqdm_postfix)
            t.update(1)

    return train_loss.avg, train_acc.avg


def test_epoch(model, device, test_loader, epoch_idx):

    # Building testing data iterator
    model.eval()
    test_loss = AverageMeter('test_loss')
    test_acc  = AverageMeter('test_acc')
    test_iter = test_loader()
    num_iters = len(test_loader)

    # Testing epoch
    with tqdm(total=num_iters,
              desc='TEST Ep. #{}'.format(epoch_idx),
              disable=False,
              dynamic_ncols=True,
              ascii=True) as t:

        for batch_idx, batch_data in enumerate(test_iter):

            images, labels = batch_data
            images = images.to(device)
            labels = labels.to(device)
            num_samples = labels.size(0)

            output = model(images)
            loss = model.module.loss_fn(output, labels)

            test_loss.update(loss.item(), num_samples)
            cls_acc = classification_accuracy(output, labels)
            test_acc.update(cls_acc, num_samples)

            tqdm_postfix = {'test_loss': round(test_loss.avg, 4),
                            'test_acc':  round(test_acc.avg, 4)}

            t.set_postfix(tqdm_postfix)
            t.update(1)

    return test_loss.avg, test_acc.avg
