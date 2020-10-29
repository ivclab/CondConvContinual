import random
import math
import numpy as np
import torchnet as tnt


class ImageDataLoader(object):
    def __init__(self, dataset, batch_size=64, zero_base_label=True, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.kwargs = kwargs

        self.categories = sorted(list(self.dataset.label2idx.keys()))
        if zero_base_label:
            self.label_mapping = {x:i for i, x in enumerate(self.categories)}
        else:
            self.label_mapping = {x:x for x in self.categories}
        return

    def build_iterator(self, epoch=0):
        rng_seed = epoch
        np.random.seed(rng_seed)
        random.seed(rng_seed)

        def load_func(index):
            image, label = self.dataset[index]
            label = self.label_mapping[label]
            return image, label

        tnt_dataset = tnt.dataset.ListDataset(
                elem_list=range(len(self.dataset.meta['image_names'])), load=load_func)

        dataloader = tnt_dataset.parallel(
                batch_size=self.batch_size, **self.kwargs)

        return dataloader

    def __call__(self, epoch=0):
        return self.build_iterator(epoch)

    def __len__(self):
        return math.ceil(len(self.dataset.meta['image_names']) / self.batch_size)
