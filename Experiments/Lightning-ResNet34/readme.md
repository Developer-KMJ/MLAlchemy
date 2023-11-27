# Simple ResNet-34 Implementation in PyTorch Lightning

This repository contains a simple implementation of ResNet-34 using PyTorch Lightning. The purpose of this implementation is just as a working ground to become familar with features of PyTorch Lightning.

## Purpose

When I decided to learn PyTorch Lightning and apply my recent training knowledge, reimplementing ResNet-34 seemed like a good challenge. 

Initially, using a standard setup and a pet dataset, my model's performance was slightly off. It achieved nearly 95% accuracy with the training set but never surpassed 44% on the validation set. This plateau occurred when it reached 95% in training, at which point its loss value was minimal. Consequently, with the training loss minimized, continued training updated the model very little, resulting in a plateau in both losses and accuracy on the validation sets.

Knowing that ResNet-34 is capable of better performance, I started by reevaluating hyperparameters and some of my design choices. Switching from Adam to AdamW as the optimizer, recalculating normalization values, experimenting with various learning rates, and incorporating label smoothing into the cross-entropy loss seemed like promising steps. Indeed, these adjustments made a significant impact. They helped increase the validation accuracy to over 60%, although the training set accuracy and losses eventually plateaued again at 95%.

Since my training loss continued to drop much faster than the validation loss, I aimed to make the model learn not memorize the training set. This led me to incorporate image augmentations. Beginning with basic techniques such as random resizing, rotation, and flipping did indeed slow down the training losses, consequently allowing the validation accuracy and loss to improve more rapidly. Therefore, I progressed to using the albumentations library for more intensive modifications like random cropping, shifting, scaling, rotating, and CoarseDropout. These efforts successfully raised the validation accuracy to the low 70s.

Given that ResNet-34 documentation and research indicate it can achieve around 82% accuracy identifying an image in a single category, I decided to implement a more drastic augmentation - MixUp. This technique combines two images in varying proportions, forcing the model to recognize categories based on the mix of each image. The model could no longer rely on memorizing images, as every epoch presented unique combinations of two images and categories, forcing it to learn patterns from data that couldn't be simply memorized. This strategy increased the validation accuracy to 83.5%. 

---

Regarding PyTorch Lightning, which was my main reason for starting this experiment, it definitely helped keep my code cleaner and eliminated redundant code. Two features I particularly appreciated were ModelCheckpoint and EarlyStopping. ModelCheckpoint was useful for keeping the top 5 models with the highest validation set accuracy. EarlyStopping proved invaluable as it automatically halted training when the validation set reached a plateau, defined as maintaining the same accuracy for 10 consecutive epochs.

## Code Overview

The code includes the definition of a `ResidualBlock` class, which represents a single residual block in the ResNet architecture. Each `ResidualBlock` consists of two convolutional layers and an optional downsampling function. The output of the block is the sum of the input (possibly downsampled) and the output of the convolutional layers.

The `ResnetTest` class is a PyTorch Lightning module that represents the entire ResNet-34 network. It includes an initial convolutional layer, followed by four sets of residual blocks, and finally a fully connected layer for classification.

The network uses the accuracy and F1 score as metrics, which are computed using the `torchmetrics` library.

## Usage

To use this code, you need to create an instance of the `ResnetTest` class and then train it using a PyTorch Lightning trainer. The number of classes and the learning rate are parameters that can be adjusted based on your specific task.