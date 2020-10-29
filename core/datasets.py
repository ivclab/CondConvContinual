from PIL import Image, ImageEnhance
import json
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast,
                       Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)


class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]
        return

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))
        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')
        return out


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def build_label2idx(labels):
    label2idx = {}
    for idx, label in enumerate(labels):
        if label not in label2idx:
            label2idx[label] = []
        label2idx[label].append(idx)
    return label2idx


class TransformManager(object):
    def __init__(self, image_size, name='standard_cifar', padding=8,
                 normalize_params = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_params    = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):

        assert(name in ['standard_cifar', 'standard_imagenet50'])
        self.image_size = image_size
        self.name = name
        self.normalize_params = normalize_params
        self.jitter_params = jitter_params
        self.padding = padding
        return

    def parse_transform(self, transform_name):

        if transform_name == 'ImageJitter':
            return ImageJitter(self.jitter_params)

        transform_cls = getattr(transforms, transform_name)
        if transform_name == 'RandomResizedCrop':
            return transform_cls(self.image_size)
        elif transform_name == 'RandomCrop':
            return transform_cls(self.image_size, padding=self.padding)
        elif transform_name == 'Normalize':
            return transform_cls(**self.normalize_params)
        elif transform_name == 'CenterCrop':
            return transform_cls(self.image_size)
        elif transform_name == 'Resize':
            return transform_cls(int(self.image_size/0.875))
        else:
            return transform_cls()

    def build_composed_transform(self, augmentation):

        if self.name == 'standard_cifar':
            if augmentation:
                transform_list = ['RandomCrop', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
            else:
                transform_list = ['ToTensor', 'Normalize']

        elif self.name == 'standard_imagenet50':
            if augmentation:
                transform_list = ['RandomResizedCrop', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
            else:
                transform_list = ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']

        else:
            raise NotImplementedError()

        transform = transforms.Compose([self.parse_transform(x) for x in transform_list])
        return transform


class ImageDataset(object):
    def __init__(self, meta_file='', transform=None, name='', meta=None):

        assert(meta_file != '' or meta is not None)

        if meta_file != '':
            with open(meta_file, 'r') as f:
                # self.meta has keys: ['image_names', 'image_labels', 'label_names']
                self.meta = json.load(f)
        else:
            self.meta = meta

        self.label2idx = build_label2idx(self.meta['image_labels'])
        self.name = name
        self.transform = transform
        return

    def __getitem__(self, index):
        image_name = self.meta['image_names'][index]
        image = self.transform(pil_loader(image_name))
        label = self.meta['image_labels'][index]
        return image, label

    def __len__(self):
        return len(self.meta['image_names'])
