import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F


MEAN = np.array([377.9830716, 653.4058739, 702.1834053, 2573.849324, 2344.568628, 1432.163695] * 3)
STD = np.array([171.5116587, 215.5407551, 364.3545396, 686.5730746, 769.6448444, 675.9192684] * 3)


class SegmentationTrainTransform:
    def __init__(self, img_size=224, mean=MEAN, std=STD):
        min_size = int(img_size * 0.6)
        max_size = int(img_size)

        trans = [
            ToTensor(),
            RandomHorizontalFlip(0.5), 
            RandomVerticalFlip(0.5),
        ]

        trans.extend(
            [
                RandomCrop(min_size, max_size),
                Resize(size=img_size),
                Clip(q=0.9),
                Normalize(mean=mean, std=std),
                ToTensor(),
            ]
        )
        self.transforms = Compose(trans)

    def __call__(self, imgs, target):
        return self.transforms(imgs, target)


class SegmentationValTransform:
    def __init__(self, mean=MEAN, std=STD):
        self.transforms = Compose(
            [
                ToTensor(),
                Normalize(mean=mean, std=std),
                ToTensor(),
            ]
        )

    def __call__(self, imgs, target):
        return self.transforms(imgs, target)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, target):
        for t in self.transforms:
            images, target = t(images, target)
        return images, target


class Resize:
    def __init__(self, size=224):
        self.size = size

    def __call__(self, images, target):
        imgs = F.resize(images, (self.size, self.size), antialias=None)
        target = target.unsqueeze(0)
        target = F.resize(target, (self.size, self.size), interpolation=transforms.InterpolationMode.NEAREST, antialias=None)
        target = target.squeeze(0)
        return imgs, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, images, target):
        imgs = images
        if random.random() < self.flip_prob:
            imgs = F.hflip(images)
            target = F.hflip(target)
        return imgs, target


class RandomVerticalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, images, target):
        imgs = images
        if random.random() < self.flip_prob:
            imgs = F.vflip(images)
            target = F.vflip(target)
        return imgs, target


class RandomCrop:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, images, target):
        H = random.randint(self.min_size, self.max_size)
        W = random.randint(self.min_size, self.max_size)
        crop_params = transforms.RandomCrop.get_params(images, (H, W))
        imgs = F.crop(images, *crop_params)
        target = F.crop(target, *crop_params)
        return imgs, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, images, target):
        imgs = [F.center_crop(image, self.size) for image in images]
        target = F.center_crop(target, self.size)
        return imgs, target


class ToTensor:
    def __call__(self, images: np.array, target):
        imgs = torch.as_tensor(images, dtype=torch.float32)
        target = torch.as_tensor(target, dtype=torch.int64)
        return imgs, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, target):
        imgs = F.normalize(images, mean=self.mean, std=self.std)
        return imgs, target


class Clip:
    def __init__(self, q=0.95):
        self.q = q

    def __call__(self, images, target):
        clip_max = np.nanquantile(images.reshape(images.shape[0], -1), self.q, axis=1)
        clip_max = np.broadcast_to(clip_max.reshape(-1, 1, 1), images.shape)
        imgs = np.clip(images, 0, clip_max)
        return imgs, target

