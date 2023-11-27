from torch import nn
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl
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

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes)

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
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, x, mixup=False):
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
        loss, scores, y = self.__common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1 = self.f1(scores, y)

        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1': f1},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):        
        loss, scores, y = self.__common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1 = self.f1(scores, y)
        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy, 'val_f1': f1},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y =  self.__common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1 = self.f1(scores, y)
        self.log_dict({'test_loss': loss, 'test_accuracy': accuracy, 'test_f1': f1},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def __common_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def configure_optimizers(self):
        #return optim.SGD(self.parameters(), lr=self.learning_rate)
        return optim.AdamW(self.parameters(), lr=self.learning_rate)

   
   
    