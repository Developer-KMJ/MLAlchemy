# Standard library imports
import argparse
import os
import time
import torch
import torch.nn as nn

# Related third party imports
from torch.optim import adamw, adam, sgd
from torch.utils.data import DataLoader

# Local application/library specific imports
from gitract_dataset import GITractDataset
from losses.combine_loss import DiceHausdorffLoss
from losses.dice import DiceLoss
# from model.test_unet import Test_UNet  

from model.unetEx import UNet

from torchvision import transforms
from torchvision.transforms import functional as F

import cv2

from datetime import datetime

# The normalization values were calculated in the prepare_images.py code
# via a call to: get_stats_for_dir()
transform = transforms.Compose([
        transforms.ToTensor(),
        ## transforms.Normalize(mean=[0.0621, 0.0621, 0.0621],
        ##                     std=[0.000559, 0.000559, 0.000559])
        ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, dataloader, optimizer, criterion, epoch, tag, output_path):
   
    save_path = os.path.join(output_path, tag)
    os.makedirs(save_path, exist_ok=True)

    epoch_path = os.path.join(output_path, f'{tag}/train/epoch{epoch}')
    os.makedirs(epoch_path, exist_ok=True)

    model.train()  # Set the model back to training mode
   
    outputs = None
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        cropped_labels = F.center_crop(labels, [outputs.shape[2], outputs.shape[3]])
    #    cv2.imwrite(f'/home/kevin/Documents/gitract/output/{j}-Label.png', cropped_labels.to('cpu').squeeze().permute(1, 2, 0).detach().numpy() * 255)
        loss = criterion(outputs, cropped_labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 50 == 49:    # print every 200 mini-batches
            orig = inputs[0].to('cpu').squeeze().permute(1, 2, 0)
            orig = orig.detach().numpy() * 255
            cv2.imwrite(f'{epoch_path}/{epoch}-{i}-Orig.png', orig)

            target = labels[0].to('cpu').squeeze().permute(1, 2, 0)
            target = target.detach().numpy() * 255
            cv2.imwrite(f'{epoch_path}/{epoch}-{i}-Target.png', target)
            
            img = outputs[0].to('cpu').squeeze().permute(1, 2, 0)
            img = img.detach().numpy() * 255
            cv2.imwrite(f'{epoch_path}/{epoch}-{i}-Image.png', img)

            print('[%d, %5d] loss: %.7f' %
                (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0

best_validation_loss = float('inf')
def validate(model, dataloader, criterion, epoch, tag, output_path):

    model.eval()  # Set the model to evaluation mode

    save_path = os.path.join(output_path, tag)
    os.makedirs(save_path, exist_ok=True)

    epoch_path = os.path.join(output_path, f'{tag}/validate/epoch{epoch}')
    os.makedirs(epoch_path, exist_ok=True)

    model_path = os.path.join(output_path, f'{tag}/savefiles')
    os.makedirs(model_path, exist_ok=True)

    with torch.no_grad():  # Temporarily turn off gradient descent
        validation_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)
            
            cropped_labels = F.center_crop(labels, [outputs.shape[2], outputs.shape[3]])
            loss = criterion(outputs, cropped_labels)

            # print statistics
            validation_loss += loss.item()

            if i % 50 == 49:    # print every 50 mini-batches
                orig = inputs[0].to('cpu').squeeze().permute(1, 2, 0)
                orig = orig.detach().numpy() * 255
                cv2.imwrite(f'{epoch_path}/{epoch}-{i}-Orig.png', orig)

                target = labels[0].to('cpu').squeeze().permute(1, 2, 0)
                target = target.detach().numpy() * 255
                cv2.imwrite(f'{epoch_path}/{epoch}-{i}-Target.png', target)
                
                img = outputs[0].to('cpu').squeeze().permute(1, 2, 0)
                img = img.detach().numpy() * 255
                cv2.imwrite(f'{epoch_path}/{epoch}-{i}-Image.png', img)

        validation_loss = validation_loss / len(dataloader)
        print('Validation loss: %.7f' % validation_loss)

        # Save the model 
        torch.save(model.state_dict(), f'{model_path}/{epoch}-{validation_loss}.pth')

        return validation_loss
        
def main():

    image_dir_list = [
            (
                "/home/kevin/Documents/gitract/data/Staged_For_Training/training/Axial/images",
                "/home/kevin/Documents/gitract/data/Staged_For_Training/training/Axial/masks",
                "/home/kevin/Documents/gitract/data/Staged_For_Training/validation/Axial/images",
                "/home/kevin/Documents/gitract/data/Staged_For_Training/validation/Axial/masks",
                "/home/kevin/Documents/gitract/output/Axial"
            ),
            (
                "/home/kevin/Documents/gitract/data/Staged_For_Training/training/Coronal/images",
                "/home/kevin/Documents/gitract/data/Staged_For_Training/training/Coronal/masks",
                "/home/kevin/Documents/gitract/data/Staged_For_Training/validation/Coronal/images",
                "/home/kevin/Documents/gitract/data/Staged_For_Training/validation/Coronal/masks",
                "/home/kevin/Documents/gitract/output/Coronal"
            ),
            (
                "/home/kevin/Documents/gitract/data/Staged_For_Training/training/Sagittal/images",
                "/home/kevin/Documents/gitract/data/Staged_For_Training/training/Sagittal/masks",
                "/home/kevin/Documents/gitract/data/Staged_For_Training/validation/Sagittal/images",
                "/home/kevin/Documents/gitract/data/Staged_For_Training/validation/Sagittal/masks",
                "/home/kevin/Documents/gitract/output/Sagittal"
            )
    ]
     
    for images in image_dir_list:
        train_image_dir = images[0]
        train_mask_dir = images[1]
        validation_image_dir = images[2]
        validation_mask_dir = images[3]
        output_path = images[4]

        train_dataset = GITractDataset(train_image_dir, train_mask_dir, transform)
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        validation_dataset = GITractDataset(validation_image_dir, validation_mask_dir, transform)
        validation_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=False)

        now = datetime.now()
        tag = now.strftime('%Y%m%d%H%M')

        model = UNet(num_classes=3)
        model = model.to(device)

        criterion = nn.MSELoss()

        optimizer = adamw.AdamW(model.parameters(), lr=0.0001)

        num_epochs = 4  # Define the number of epochs
        for epoch in range(num_epochs):
            train(model, train_dataloader, optimizer, criterion, epoch, tag, output_path)
            time.sleep(60)
            validate(model, validation_dataloader, criterion, epoch, tag, output_path)
            time.sleep(60)

def original_main():

    args = parse_args()

    train_image_dir = args.train_image_dir
    train_mask_dir = args.train_mask_dir
    train_dataset = GITractDataset(train_image_dir, train_mask_dir, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    validation_image_dir = args.validation_image_dir
    validation_mask_dir = args.validation_mask_dir
    validation_dataset = GITractDataset(validation_image_dir, validation_mask_dir, transform)
    validation_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=False)
    
    output_path = args.output_path

    # model = UNet(num_classes=3)
    # save_state = torch.load('kj-best_model-loss1.647.pth')
    # model.load_state_dict(save_state)
    # model = Test_UNet(3, 3)

    now = datetime.now()
    tag = now.strftime('%Y%m%d%H%M')

    model = UNet(num_classes=3)
    model = model.to(device)

    criterion = nn.MSELoss() # .CrossEntropyLoss() # .MSELoss()
    # criterion = DiceHausdorffLoss()
    # criterion = DiceLoss()
    optimizer = adamw.AdamW(model.parameters(), lr=0.0001)
    # optimizer = sgd.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, threshold=.0001, verbose=True)

    num_epochs = 4  # Define the number of epochs
    for epoch in range(num_epochs):
        train(model, train_dataloader, optimizer, criterion, epoch, tag, output_path)
        time.sleep(60)
        validate(model, validation_dataloader, criterion, epoch, tag, output_path)
        time.sleep(60)
        # scheduler.step(loss_metric)

    print('Finished Training')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="test_main.py",
        description="Convert csv file to segmented image.")
    
    parser.add_argument("--train_image_dir", type=str, help="Path to training image data", required=True)
    parser.add_argument("--train_mask_dir", type=str, help="Path to training mask data.", required=True)
    parser.add_argument("--validation_image_dir", type=str, help="Path to validation image data.", required=True)
    parser.add_argument("--validation_mask_dir", type=str, help="Path to validation mask data.", required=True)
    parser.add_argument("--output_path", type=str, help="Location for output", required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
    


'''

def train(model, dataloader, optimizer, criterion, epoch):
    model.train()  # Set the model back to training mode
    
    for j in range(100000):
        outputs = None
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            cropped_labels = F.center_crop(labels, outputs.shape[2])
        #    cv2.imwrite(f'/home/kevin/Documents/gitract/output/{j}-Label.png', cropped_labels.to('cpu').squeeze().permute(1, 2, 0).detach().numpy() * 255)
            loss = criterion(outputs, cropped_labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

        if j % 200 == 199:
            img = outputs.to('cpu').squeeze().permute(1, 2, 0)
            img = img.detach().numpy() * 255
            cv2.imwrite(f'/home/kevin/Documents/gitract/output/{j}-Image.png', img)
            
'''
