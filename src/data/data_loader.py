import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LungDataset(Dataset):
    def __init__(self, dataset_name, config, transform=None):
        self.dataset_name = dataset_name.lower()
        self.transform = transform
        self.config = config
        if self.dataset_name not in self.config['datasets']:
            raise ValueError(f"Dataset '{dataset_name}' not defined in config.yaml.")
        self.image_files = self._find_image_files()

    def _find_image_files(self):
        dataset_info = self.config['datasets'][self.dataset_name]
        image_dir = os.path.join(dataset_info['path'], dataset_info['images'])
        return sorted(glob.glob(os.path.join(image_dir, "*.png")))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        base_name = os.path.basename(img_path)
        image = Image.open(img_path).convert("L")

        dataset_info = self.config['datasets'][self.dataset_name]
        mask_base_path = os.path.join(dataset_info['path'], dataset_info['masks'])

        if self.dataset_name == 'montgomery':
            left_mask_path = os.path.join(mask_base_path, "leftMask", base_name)
            right_mask_path = os.path.join(mask_base_path, "rightMask", base_name)
        else:  # jsrt
            left_mask_path = os.path.join(mask_base_path, "left_lung", base_name)
            right_mask_path = os.path.join(mask_base_path, "right_lung", base_name)

        left_mask = np.array(Image.open(left_mask_path).convert("L"))
        right_mask = np.array(Image.open(right_mask_path).convert("L"))
        mask_np = np.maximum(left_mask, right_mask)

        mask = Image.fromarray(mask_np)
        sample = {'image': image, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ResizeAndToTensor(object):
    def __init__(self, size=(256, 256)):
        self.size = size
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = self.normalize(self.to_tensor(image.resize(self.size, Image.BILINEAR)))
        mask = (self.to_tensor(mask.resize(self.size, Image.NEAREST)) > 0.5).float()
        return {'image': image, 'mask': mask}
