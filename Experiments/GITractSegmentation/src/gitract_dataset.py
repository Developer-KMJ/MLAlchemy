import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2

class GITractDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = os.path.join(self.image_dir, self.images[idx])
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
      
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
      
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


class GITractValidationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = os.path.join(self.image_dir, self.images[idx])
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, self.images[idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        else:
            mask = None
      
        if self.transform:
            image = self.transform(image)
            
            if mask is not None:
                mask = self.transform(mask)
            
        return image, mask, img_path, mask_path
