import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Metric
from torch.autograd import Variable
import torchmetrics

from config import *

   
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample_fn=None):
        super(ResidualBlock, self).__init__()
       
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.downsample_fn = downsample_fn

        self.relu = nn.ReLU(inplace=True)

        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample_fn is not None:
            residual = self.downsample_fn(residual)
        x += residual
        x = self.relu(x)
        return x


class ResnetTest(pl.LightningModule):
    # Same as the init from an NN.Module
    def __init__(self, num_classes):
        super(ResnetTest, self).__init__()

        # Adding to find learning rate finder.
        self.learning_rate = LEARNING_RATE
        self.mixup_probability = MIXUP_PROBABILITY

        self.in_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )

        self.block2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, downsample_fn=nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(128)
            )),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
        )

        self.block3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, downsample_fn=nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(256)
            )),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
        )

        self.block4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2, downsample_fn=nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(512)
            )),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, num_classes)

        # Trying label smoothing to see if I can improve the accuracy
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)

       #  self.metric = torchmetrics.functional.accuracy

        self.metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

        self.mix_up = True
        self.alpha = 1.0
    
    def forward(self, x):
        x = self.in_layer(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def training_step(self, batch, batch_idx):
        # loss, accuracy = self.__shared_step(batch, batch_idx, is_train=True)
        loss, accuracy = self.__shared_step(batch, batch_idx, is_train=False)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):        
        loss, accuracy = self.__shared_step(batch, batch_idx, is_train=False)
        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def test_step(self, batch, batch_idx):
        loss, accuracy = self.__shared_step(batch, batch_idx, is_train=False)
        self.log_dict({'test_loss': loss, 'test_accuracy': accuracy},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)

    def __mixup_data(self, x, y, alpha=1.0):
        '''Returns mixed inputs, targets, and lambda
        Parameters
        ----------
        x: input data
        y: target
        alpha: value of alpha and beta in beta distribution 
        '''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size) # shuffle index

        mixed_x = lam * x + (1 - lam) * x[index, :] # mixup between original image order and shuffled image order
        y_a, y_b = y, y[index] # return target of both images order
        
        return mixed_x, y_a, y_b, lam

    def __mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * self.loss(pred, y_a) + (1 - lam) * self.loss(pred, y_b)
    
    def __mixup_accuracy(self, preds, y_a, y_b, lam):
        """
        Updated metric calculation:
        Args:
        -----
        metric: metric to use, example: accuracy
        preds: predictions from network
        y_a: original labels
        y_b: labels of the shuffled batch
        lam: alpha used for mixup
        """
        return lam * self.metric(preds, y_a) + (1 - lam) * self.metric(preds, y_b)
    
    def __shared_step(self, batch, batch_idx, is_train=False):
        x, y = batch
        if is_train and self.mix_up: # if mixup is true and train
            # prepare the mixup date
            x, y_a, y_b, lam = self.__mixup_data(x, y, self.alpha)
            x, y_a, y_b = map(Variable, (x, y_a, y_b))
            # pass the new data through model
            logits = self(x)
            # calculate loss
            loss = self.__mixup_criterion(logits, y_a, y_b, lam)
            # calculate accuracy
            preds = torch.argmax(logits, dim=1)
            acc = self.__mixup_accuracy(preds, y_a, y_b, lam)
        else: # if mixup is false or validation
            # no change in data, we padd the batch data as is
            # pass the data through model
            logits = self(x)
            # calculate loss
            loss = self.loss(logits, y)
            # calculate accuracy
            preds = torch.argmax(logits, dim=1)
            acc = self.metric(preds, y)
        
        return loss, acc