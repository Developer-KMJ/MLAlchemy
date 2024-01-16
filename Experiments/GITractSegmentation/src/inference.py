# Standard library imports
import os
import numpy as np
import torch
import torch.nn as nn
import cv2

# Related third party imports
from torch.optim import adamw
from torch.utils.data import DataLoader

# Local application/library specific imports
from gitract_dataset import GITractValidationDataset

from model.unetEx import UNet

from torchvision import transforms
from torchvision.transforms import functional as F

from torchmetrics.classification import Dice

from kornia.losses import HausdorffERLoss3D, DiceLoss

# The normalization values were calculated in the prepare_images.py code
# via a call to: get_stats_for_dir()
transform = transforms.Compose([
        transforms.ToTensor()])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    test_inference()
    

def test_inference():
    validation_image_dir = '/home/kevin/Documents/gitract/scratch/Image'
    validation_mask_dir = '/home/kevin/Documents/gitract/scratch/Mask'

    output_dir = '/home/kevin/Documents/gitract/output/loss-test'

    out_working_mask_dir = f'{output_dir}/working-mask'
    os.makedirs(out_working_mask_dir, exist_ok=True)

    out_actual_mask_dir = f'{output_dir}/actual-mask'
    os.makedirs(out_actual_mask_dir, exist_ok=True)

    out_predicted_mask_dir = f'{output_dir}/predicted-mask'
    os.makedirs(out_predicted_mask_dir, exist_ok=True)

    # Create the dataset and dataloader
    validation_dataset = GITractValidationDataset(validation_image_dir, validation_mask_dir, transform)
    validation_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=False)

    # Load the model
    model = UNet(num_classes=3)
    save_state = torch.load('/home/kevin/Documents/gitract/output/Axial/202401112027/savefiles/3-0.0018428458184015328.pth')
    model.load_state_dict(save_state)

    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    dice = Dice(threshold=0.6, average='micro').to(device)
    accumulated_dice_losses = 0

    with torch.no_grad():  # Temporarily turn off gradient descent
        for i, data in enumerate(validation_dataloader, 0):
            inputs, actual_mask, image_path, mask_path = data
            inputs = inputs.to(device)
            actual_mask = actual_mask.to(device).to(dtype=torch.int)
            
            # Prediction
            predicted_mask = model(inputs)
            predicted_mask = F.center_crop(predicted_mask, [inputs.shape[2], inputs.shape[3]])

            image = predicted_mask.cpu()
            images = torch.split(image, 1, dim=0)

            resized_masks = []
            for j, image in enumerate(images):
                image = image.squeeze() \
                             .detach() \
                             .permute(1, 2, 0) \
                             .numpy()
                image = image * 255
                
                cv2.imwrite(f'{out_working_mask_dir}/{os.path.basename(mask_path[j])}', image)

            # Calculate the loss using dice here. 
            accumulated_dice_losses += dice(predicted_mask, actual_mask)

        # Calculate the average loss
        average_dice_loss = accumulated_dice_losses / len(validation_dataloader)
        average_dice_score = 1 - average_dice_loss

    # Calculate the score using hausdorff here.
    # Hausdorff uses a 3d image. So we have to put all the pieces together. 
    hausdorff_score = calculate_hausdorff_score(validation_mask_dir,
                                                out_working_mask_dir,
                                                out_actual_mask_dir,
                                                out_predicted_mask_dir)
    
    print(f'Hausdorff Score: {hausdorff_score}')
    print(f'Dice Score: {average_dice_score}')

    # The context mixes the scores on the basis of 60% hausdorff and 40% dice
    print(f'Mixed Loss: {(0.6 * hausdorff_score) + (0.4 * average_dice_score)}')





def calculate_dice_score(predicted, actual):
    dice_loss = DiceLoss()

     # Calculate the loss using dice here. 
    np_predicted = np.array(predicted)
    # np_predicted = np.where(np_predicted > 0.6, 1, 0) 
    local_predicted = torch.from_numpy(np_predicted).to(dtype=torch.float32)

    np_actual = np.array(actual)
    np_actual_max = np.max(np_actual, axis=1)
    np_actual = np.where(np_actual_max > 0, np.argmax(np_actual, axis=1) + 1, 0)
    local_actual = torch.from_numpy(np_actual).to(dtype=torch.long)
    
    return (1 - dice_loss(local_predicted, local_actual))


def calculate_hausdorff_score(validation_mask_dir: str, 
                             out_working_mask_dir: str, 
                             out_actual_mask_dir: str, 
                             out_predicted_mask_dir: str):
    # Put masks on disk temporarily so we can see them, we will read from these 
    # directories to create the 3d masks for comparison. 
    create_masks_from_image(validation_mask_dir, out_actual_mask_dir, 1.0)
    create_masks_from_image(out_working_mask_dir, out_predicted_mask_dir, 0.5)

    actual_3d = create_3d_mask(os.path.join(out_actual_mask_dir, "Stomach"))
    predicted_3d = create_3d_mask(os.path.join(out_predicted_mask_dir, "Stomach"))

    loss = HausdorffERLoss3D()

    actual_torch_3d = torch.from_numpy(actual_3d)
    predicted_torch_3d = torch.from_numpy(predicted_3d)
    
    # Requires batch, channels, depth, height, width. 
    # batch in this case is just 1 image
    # channels is just 1 channel as we are grayscale
    # depth is the number of slices in the image
    # height is the height of the image
    # width is the width of the image
    actual_torch_3d = actual_torch_3d.unsqueeze(0).unsqueeze(0)
    predicted_torch_3d = predicted_torch_3d.unsqueeze(0).unsqueeze(0)

    loss_score = loss(predicted_torch_3d, 
                      actual_torch_3d)
    
    # normalize loss_score by dividing by width * height
    # the depth for each slice is 1, so we don't need to divide by that.
    
    # Measures the degree of similarity between two Geometrys using the Hausdorff distance metric. 
    # The measure is normalized to lie in the range [0, 1]. Higher measures indicate a great degree of similarity.
    # The measure is computed by computing the Hausdorff distance between the input geometries, and then normalizing 
    # this by dividing it by the diagonal distance across the envelope of the combined geometries.

    diagonal = np.sqrt(actual_torch_3d.shape[3]**2 + actual_torch_3d.shape[4]**2)
    loss_score = loss_score / diagonal


    # Although HD is used extensively in evaluating the segmentation performance,
    # segmentation algorithms rarely aim at minimizing or reducing HD directly

    # Based on this statement above, the lower the measure the better. But since
    # is it is being used as a score, we will return the inverse of the measure
    return 1 - loss_score

def create_masks_from_image(mask_dir: str, 
                            out_dir: str,
                            threshold: float = 1):
    
    def create_files(out_dir: str, postfix: str, file: str, mask: np.array):
        _path = f'{out_dir}/{postfix}'
        os.makedirs(_path, exist_ok=True)
        _file = os.path.join(_path, os.path.basename(file))
        cv2.imwrite(_file, mask * 255)
        # _file = _file.replace('.png', '.npy')
        # np.save(_file, mask)
                             
    for file in sorted(os.listdir(mask_dir)):
        # Create Red-Stomach, Green-Large_Bowel, Blue-Small_Bowel  
        # into one-bit-encodings for the comparision. 

        mask_img = cv2.imread(os.path.join(mask_dir, file), cv2.IMREAD_UNCHANGED)
        mask_img = mask_img / 255.0
        mask_img = np.where(mask_img >= threshold, 1, 0)
        si_blue_mask = mask_img[:,:,0]
        li_green_mask = mask_img[:,:,1]
        stomach_red_mask = mask_img[:,:,2]

        create_files(out_dir, 'SI', file, si_blue_mask)
        create_files(out_dir, 'LI', file, li_green_mask)
        create_files(out_dir, 'Stomach', file, stomach_red_mask)

def create_3d_mask(mask_dir: str):
    mask_list = []
    for file in sorted(os.listdir(mask_dir)):
        mask = cv2.imread(os.path.join(mask_dir, file), cv2.IMREAD_UNCHANGED)
        mask_list.append(mask)
    
    mask_list = np.array(mask_list) /255.0
    return mask_list

if __name__ == "__main__":
    main()









        

       
 # imageList = []
    # for file in sorted(os.listdir(validation_mask_dir)):
    #     # Create Red-Stomach, Green-Large_Bowel, Blue-Small_Bowel  
    #     # into one-bit-encodings for the comparision. 
    #     mask_img = cv2.imread(os.path.join(validation_mask_dir, file), cv2.IMREAD_UNCHANGED)
    #     # mask_img = np.transpose(mask_img, (2, 0, 1))
    #     mask_img = mask_img / 255.0
    #     mask_img = np.where(mask_img == 1, 1, 0)
    #     si_blue_mask = mask_img[:,:,0]
    #     li_green_mask = mask_img[:,:,1]
    #     stomach_red_mask = mask_img[:,:,2]

    #     np.save(f'{out_actual_si_dir}/{os.path.basename(file)}', si_blue_mask)
    #     np.save(f'{out_actual_li_dir }/{os.path.basename(file)}', li_green_mask)
    #     np.save(f'{out_actual_stomach_dir}/{os.path.basename(file)}', stomach_red_mask)


    #     # image = [li_green_mask, li_green_mask, li_green_mask]
    #     # image = np.transpose(image, (1, 2, 0))
    #     # image = np.array(image)
    #     # image = image * 255

    #     # cv2.imwrite(f'{out_actual_stomach_dir}/{os.path.basename(file)}', image)

