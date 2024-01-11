# Standard library imports
import numpy as np
import torch
import torch.nn as nn
import cv2

# Related third party imports
from torch.optim import adamw
from torch.utils.data import DataLoader

# Local application/library specific imports
from gitract_dataset import GITractDataset, GITractValidationDataset  

from model.unetEx import UNet

from torchvision import transforms
from torchvision.transforms import functional as F

# The normalization values were calculated in the prepare_images.py code
# via a call to: get_stats_for_dir()
transform = transforms.Compose([
        transforms.ToTensor(),
        ## transforms.Normalize(mean=[0.0621, 0.0621, 0.0621],
        ##                     std=[0.000559, 0.000559, 0.000559])
        ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    test_inference()
    

def test_inference():


    validation_image_dir = '/home/kevin/Documents/gitract/scratch/inputImg'
    validation_dataset = GITractValidationDataset(validation_image_dir, transform)
    validation_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=False)

    criterion = nn.CrossEntropyLoss() # .MSELoss()

    # Load the model
    model = UNet(num_classes=3)
    save_state = torch.load('model-173-0.005733715236486717.pth')
    model.load_state_dict(save_state)

    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Temporarily turn off gradient descent
        validation_loss = 0.0
        for i, data in enumerate(validation_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data
            inputs = inputs.to(device)
            
            # forward
            outputs = model(inputs)
            
            input_image = inputs.cpu()
            input_images = torch.split(input_image, 1, dim=0)
            for j, image in enumerate(input_images):
                image = image.squeeze()
                image = image.permute(1, 2, 0)
                image = image.numpy() * 255
                image = image.astype('uint8')
                cv2.imwrite(f'/home/kevin/Documents/gitract/output/{i * 16 + j}-Image.png', image)

            image = outputs.cpu()
            images = torch.split(image, 1, dim=0)

            for j, image in enumerate(images):
                image = image.squeeze()
                image = image.numpy() * 255
                image = image.astype('uint8')

                image = np.transpose(image, (1, 2, 0))
                
                cv2.imwrite(f'/home/kevin/Documents/gitract/output/{i * 16 + j}-Mask.png', image)

if __name__ == "__main__":
    main()