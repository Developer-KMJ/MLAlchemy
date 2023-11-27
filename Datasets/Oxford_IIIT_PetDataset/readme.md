# Oxford-IIIT Pet Dataset Downloader

## Overview
This directory contains code used for downloading and preparing the Oxford-IIIT Pet Dataset for training with PyTorch Lightning. The code includes a custom data module, `PetsDataModule`, for handling data preprocessing, augmentation, and setup of training, validation, and test data loaders.

## Features
- Download and extract the Oxford-IIIT Pet Dataset.
- Sort images into appropriate directories based on their names.
- Implement image augmentations using `albumentations` library.
- Use PyTorch Lightning's `LightningDataModule` for efficient data handling.
- Configure data loaders for training, validation, and test sets.

## Requirements
- PyTorch Lightning
- torchvision
- PIL
- numpy
- albumentations
- Python 3.x

## Usage

1. **Data Preparation**:
   The `PetsDataModule` class handles data preparation, including downloading and extracting the dataset, sorting images, and splitting them into train, validation, and test sets based on specified percentages.

2. **Data Augmentation**:
   Image augmentations are implemented using the `albumentations` library for the training dataset, and standard torchvision transforms for the validation and test datasets.

3. **Data Loaders**:
   The module provides separate data loaders for training, validation, and test datasets, with options to customize batch size and number of workers.

4. **Configuration**:
   Modify `config.py` to set parameters like batch size, number of workers, split percentages, and path to the dataset.

## Example
```python
data_module = PetsDataModule(data_dir="path/to/data", batch_size=32, num_workers=4)
data_module.prepare_data()
data_module.setup()
train_loader = data_module.train_dataloader()
```

## Note
Ensure that the Oxford-IIIT Pet Dataset is available in the specified `data_dir` before running the module. The dataset can be downloaded from [here](https://www.robots.ox.ac.uk/~vgg/data/pets/).

## Contributions
Contributions to this project are welcome. Please ensure that any pull requests maintain the code standards and improve functionality or efficiency.