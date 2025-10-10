import os
import torch
import numpy as np
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    image_dir = os.path.join(path, 'train')

    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(256),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    # Choose dataset implementation based on folder layout:
    # 1) Flat layout: <data_root>/train/{blur,sharp}
    # 2) Hierarchical layout: <data_root>/train/<scene>/{blur,sharp}
    if os.path.isdir(os.path.join(image_dir, 'blur')) and os.path.isdir(os.path.join(image_dir, 'sharp')):
        dataset = DeblurDataset(image_dir, transform=transform)
    else:
        dataset = HierarchicalDeblurDataset(image_dir, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'valid')),  # os.path.join(path, 'valid')=dataset/GOPRO/valid/
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'blur/'))  # 模糊图片  image_dir=dataset/GOPRO/valid/blur/
        self._check_image(self.image_list)  # 检查图片的格式
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'blur', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'sharp', self.image_list[idx]))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError


class HierarchicalDeblurDataset(Dataset):
    """
    Train dataset that supports multiple scene subfolders under <train>.
    Expected layout:
      <data_root>/train/<scene>/{blur,sharp}/<filename>
    Only used for training; valid/test keep using DeblurDataset.
    """

    def __init__(self, train_root, transform=None):
        self.transform = transform
        self.pairs = []  # list of (blur_path, sharp_path)

        if not os.path.isdir(train_root):
            raise ValueError('Train root not found: %s' % train_root)

        # Iterate scene folders
        for scene in sorted(os.listdir(train_root)):
            scene_path = os.path.join(train_root, scene)
            if not os.path.isdir(scene_path):
                continue
            blur_dir = os.path.join(scene_path, 'blur')
            sharp_dir = os.path.join(scene_path, 'sharp')
            if not (os.path.isdir(blur_dir) and os.path.isdir(sharp_dir)):
                continue

            img_list = [f for f in os.listdir(blur_dir) if f.split('.')[-1].lower() in ['png', 'jpg', 'jpeg']]
            img_list.sort()
            for name in img_list:
                blur_path = os.path.join(blur_dir, name)
                sharp_path = os.path.join(sharp_dir, name)
                if os.path.isfile(blur_path) and os.path.isfile(sharp_path):
                    self.pairs.append((blur_path, sharp_path))

        if len(self.pairs) == 0:
            raise ValueError('No training images found under %s. Ensure structure is train/<scene>/{blur,sharp}.' % train_root)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        blur_path, sharp_path = self.pairs[idx]
        image = Image.open(blur_path)
        label = Image.open(sharp_path)

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        return image, label
