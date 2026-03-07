"""
Dataset and data loading utilities for ASL classification
"""

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


class ASLDataset(Dataset):
    """
    ASL Dataset wrapper around ImageFolder

    Args:
        data_dir (str): Path to dataset directory
        transform (callable, optional): Transform to apply to images
    """
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        """Return list of class names"""
        return self.data.classes


def get_transforms(image_size=128, is_training=True, augmentation_config=None):
    """
    Get data transforms for training or validation

    Args:
        image_size (int): Size to resize images to
        is_training (bool): Whether transforms are for training
        augmentation_config (dict, optional): Augmentation configuration

    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    # ImageNet normalization (matches EfficientNet pretrained expectations)
    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if is_training and augmentation_config and augmentation_config.get('enabled', False):
        transform_list = [
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomCrop(image_size),
        ]

        if augmentation_config.get('horizontal_flip', False):
            transform_list.append(transforms.RandomHorizontalFlip())

        if 'rotation_degrees' in augmentation_config:
            transform_list.append(
                transforms.RandomRotation(augmentation_config['rotation_degrees'])
            )

        if 'color_jitter' in augmentation_config:
            transform_list.append(
                transforms.ColorJitter(**augmentation_config['color_jitter'])
            )

        if augmentation_config.get('random_perspective', False):
            transform_list.append(
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3)
            )

        if augmentation_config.get('random_affine', False):
            transform_list.append(
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
            )

        if augmentation_config.get('gaussian_blur', False):
            transform_list.append(
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            )

        transform_list.extend([transforms.ToTensor(), imagenet_normalize])

        if augmentation_config.get('random_erasing', False):
            transform_list.append(
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
            )
    else:
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            imagenet_normalize,
        ]

    return transforms.Compose(transform_list)
