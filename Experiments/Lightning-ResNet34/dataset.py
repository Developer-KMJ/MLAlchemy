import os
import random
import re
import shutil
import tarfile

import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from PIL import Image
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

import config as Config

class AlbumentationsImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, loader=Image.open):
        super().__init__(root, loader, extensions=('jpg', 'jpeg', 'png', 'bmp'), transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(image=np.array(sample))['image']
        return sample, target
    

class PetsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int) -> None:
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        super().__init__()

    def get_num_classes(self) -> int:
        if os.path.exists(os.path.join(self.data_dir, 'train')):
            # The data has already been prepared
            return len(os.listdir(os.path.join(self.data_dir, 'train')))
        else:
            self.prepare_data()
            return len(os.listdir(os.path.join(self.data_dir, 'train')))
        
    def prepare_data(self) -> None:
        # Single GPU even on multi gpu systems.
        # This is where you want to download the data, since it
        # is being done once.
        
        # This example uses: https://www.robots.ox.ac.uk/~vgg/data/pets/
        # The data was preloaded into the data/raw folder as tar.gz files.

        if os.path.exists(os.path.join(self.data_dir, 'train')):
            # The data has already been prepared
            return

        # Extract the tar.gz files
        compressed_images = os.path.join(self.data_dir, 'raw', 'images.tar.gz')
        
        with tarfile.open(compressed_images, "r:gz") as tar:
            tar.extractall(self.data_dir)
        
        # images are stored in a subfolder from the gz called images
        uncompressed_image_dir = os.path.join(self.data_dir, 'images')
        self.__sort_folder_structure(uncompressed_image_dir)
       
        # assert (
        #     Config.TRAIN_SPLIT_PERCENT + Config.VAL_SPLIT_PERCENT + Config.TEST_SPLIT_PERCENT <= 1,
        #     "The sum of TRAIN_SPLIT_PERCENT, VAL_SPLIT_PERCENT, and TEST_SPLIT_PERCENT should be less than or equal to 1")
        
        self.__sample_images(
            percent_train=Config.TRAIN_SPLIT_PERCENT,
            percent_val=Config.VAL_SPLIT_PERCENT,
            percent_test=Config.TEST_SPLIT_PERCENT)
        
        # Count the number of classes
        self.num_classes = len(os.listdir(os.path.join(self.data_dir, 'train')))


    def setup(self, stage: str = None) -> None:
        # If using Multiple, it will run on all gpus.
        # Build the dataset
        self.__create_datasets()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          pin_memory=True)
    
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          pin_memory=True)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          pin_memory=True)
 
    def __sort_folder_structure(self, image_dir: str):
        # Move the images into directories based on their name
        for path in os.listdir(image_dir):

            source_file = os.path.join(image_dir, path)
            # Skip directories
            if os.path.isdir(source_file):
                continue

            # Move images into directories based on their name
            # Categorizing this way allows us to use the ImageFolder dataset
            # later.
            base_name = os.path.basename(path)  # Get the filename with extension
            filename_pieces = os.path.splitext(base_name)
            file_name_without_extension = filename_pieces[0]  # Remove the extension
            file_name_without_extension = re.sub(r'_[0-9]+$', '', file_name_without_extension)  # Remove _<number> at the end
            target_path = os.path.join(image_dir, file_name_without_extension)
            if (not os.path.exists(target_path)):
                os.makedirs(target_path)

            if filename_pieces[1] != '.jpg':
                os.remove(source_file)
            else:
                # Move the file into a directory that matches the category name.
                shutil.move(source_file, os.path.join(target_path, base_name))

    def __sample_images(self, percent_train: float, percent_val: float, percent_test: float):
        # Sample the images into train, val, and test folders
        # This is done after the images are sorted into folders based on their name.
        # This is done so that the train, val, and test sets have the same distribution
        # of images.
        image_dir = os.path.join(self.data_dir, 'images')
        for category_name in os.listdir(image_dir):
            category_dir = os.path.join(image_dir, category_name)
            if not os.path.isdir(category_dir):
                continue

            # Get all the images in the category
            images = os.listdir(category_dir)
            num_images = len(images)

            # Sample the images
            num_train = int(num_images * percent_train)
            num_val = int(num_images * percent_val)
            num_test = int(num_images * percent_test)

            # Sample without replacement
            train_images = random.sample(images, num_train)
            images = [image for image in images if image not in train_images]
            val_images = random.sample(images, num_val)
            test_images = [image for image in images if image not in val_images]

            # Move the images into the train, val, and test directories
            self.__move_images(train_images, category_dir, category_name, 'train')
            self.__move_images(val_images, category_dir, category_name,  'val')
            self.__move_images(test_images, category_dir, category_name, 'test')
            os.rmdir(category_dir)
        
        os.rmdir(image_dir)

    def __move_images(self, 
                      images: list[str], 
                      source_dir: str, 
                      category: str, 
                      target_dir: str):
        
        # Move the images from the source directory to the target directory
        target_dir = os.path.join(self.data_dir, target_dir, category)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        for image in images:
            source_file = os.path.join(source_dir, image)
            target_file = os.path.join(target_dir, image)
            shutil.move(source_file, target_file)

    def __create_datasets(self):

        # For resnet training the images need to be transformed:
        #   - 224x224
        #   - Normalized with mean and std of ImageNet
        # augmented_transform = transforms.Compose([
        #         transforms.RandomResizedCrop(224),
        #         transforms.RandomRotation(10),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.4846, 0.4492, 0.3971),
        #                             (0.2631, 0.2581, 0.2657))])

        augmented_transform = A.Compose([
                A.RandomResizedCrop(224, 224),
                A.HorizontalFlip(p=0.3),
                A.CoarseDropout(p=0.3),
                A.PixelDropout(p=0.3),
                A.ShiftScaleRotate(p=0.3),
                A.Normalize(mean=(0.4846, 0.4492, 0.3971), std=(0.2631, 0.2581, 0.2657)),
                ToTensorV2()])

        standard_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4846, 0.4492, 0.3971),
                                    (0.2631, 0.2581, 0.2657))])

        train_dir = os.path.join(self.data_dir, 'train')
        self.train_dataset = AlbumentationsImageFolder(train_dir, transform=augmented_transform)
        
        val_dir = os.path.join(self.data_dir, 'val')
        self.val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=standard_transform)
        
        test_dir = os.path.join(self.data_dir, 'test')
        self.test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=standard_transform)
