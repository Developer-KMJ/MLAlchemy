# Standard library imports
from datetime import datetime
import os
from typing import Any, Tuple

# Related third party imports
import cv2
import numpy as np
from scipy.spatial.distance import cdist
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import HausdorffERLoss3D, HausdorffERLoss
from torchmetrics import Dice
from torchvision.transforms import functional as TF

# Local application/library specific imports
from common import LossMetric, TrainingStage, config

class UNet(pl.LightningModule):
    def __init__(self, num_classes, tag = None):
        super(UNet, self).__init__()

        self.criterion = nn.MSELoss()
        self.dice = Dice(threshold=0.6, average='micro')
        self.hausdorff = HausdorffERLoss3D()

        self.num_classes = num_classes
        self.write_intermediate = False 
             
        if tag is None:  
            self.tag = datetime.now().strftime('%Y%m%d%H%M')
        else:
            self.tag = tag
        
      # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3
        self.down_block_1 = UNet._double_conv_block(3, 64) # 572x572x3 -> 570x570x64 -> 568x568x64 -> 284x284x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input: 284x284x64
        self.down_block_2 = UNet._double_conv_block(64, 128) # 284x284x64 -> 282x282x128 -> 280x280x128 -> 140x140x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)        
        
        # input: 140x140x128
        self.down_block_3 = UNet._double_conv_block(128, 256) # 140x140x128 -> 138x138x256 -> 136x136x256 -> 68x68x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input: 68x68x256
        self.down_block_4 = UNet._double_conv_block(256, 512) # 68x68x256 -> 66x66x512 -> 64x64x512 -> 32x32x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck
        self.bottleneck = UNet._double_conv_block(512, 1024) # 32x32x512 -> 30x30x1024 -> 28x28x1024

        # Decoder
        self.up_conv_4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_block_4 = UNet._double_conv_block(1024, 512)
       
        self.up_conv_3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_block_3 = UNet._double_conv_block(512, 256)

        self.up_conv_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_block_2 = UNet._double_conv_block(256, 128)
        
        self.up_conv_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_block_1 = UNet._double_conv_block(128, 64)
       
        # Output layer
        self.outconv = nn.Conv2d(64, num_classes, kernel_size=1, padding='same')
        
        # for p in self.parameters():
        #     print(f'min:{p.min()}')
        #     print(f'max(p):{p.max()}')
        #     # torch.nn.init.uniform_(p, 0, 1)

    def set_write_intermediate(self, write_intermediate):
        self.write_intermediate = write_intermediate

    def forward(self, x):
        # Encoder

        # Block 1
        x = self.down_block_1(x)
        b1_residual = x
        x = self.pool1(x)
        
        x = self.down_block_2(x)
        b2_residual = x
        x = self.pool2(x)

        x = self.down_block_3(x) 
        b3_residual = x
        x = self.pool3(x)

        x = self.down_block_4(x)
        b4_residual = x
        x = self.pool4(x)
        
        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up_conv_4(x)
        cropped_b4_residual = UNet._crop_to_match_target(b4_residual, x)
        x = torch.cat([x, cropped_b4_residual], dim=1)
        x = self.up_block_4(x)

        x = self.up_conv_3(x)
        cropped_b3_residual = UNet._crop_to_match_target(b3_residual, x)
        x = torch.cat([x, cropped_b3_residual], dim=1)
        x = self.up_block_3(x)

        x = self.up_conv_2(x)
        cropped_b2_residual = UNet._crop_to_match_target(b2_residual, x)
        x = torch.cat([x, cropped_b2_residual], dim=1)
        x = self.up_block_2(x)

        x = self.up_conv_1(x)
        cropped_b1_residual = UNet._crop_to_match_target(b1_residual, x)
        x = torch.cat([x, cropped_b1_residual], dim=1)
        x = self.up_block_1(x)
    
        # Output layer
        out = self.outconv(x)

        return out

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, str, str], batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, batch_idx, LossMetric.TRAIN_LOSS, TrainingStage.TRAIN, self.write_intermediate)
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, str, str], batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, batch_idx, LossMetric.VAL_LOSS, TrainingStage.VALIDATION, self.write_intermediate)
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, str, str], batch_idx: int) -> torch.Tensor:
        return self._dice_hausdorff_step(batch, batch_idx, LossMetric.VAL_LOSS, TrainingStage.VALIDATION, self.write_intermediate)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        return optimizer
        

    @staticmethod
    def _double_conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.ReLU(True),
        )
       
    @staticmethod    
    def _crop_to_match_target(residual, target):

        e1 = (int)((residual.shape[2] - target.shape[2])/2)
        e2 = (int)(residual.shape[2] - ((residual.shape[2] - target.shape[2]) - e1))

        e3 = (int)((residual.shape[3] - target.shape[3])/2)
        e4 = (int)(residual.shape[3] - ((residual.shape[3] - target.shape[3]) - e3))

        cropped_residual = residual[:, :, e1:e2, e3:e4]
        return cropped_residual


    def _common_step(self,  
                     batch: Tuple[torch.Tensor, torch.Tensor, str, str],  
                     batch_idx: int,
                     metric_name : LossMetric,
                     stage: str = TrainingStage.TRAIN,
                     write_intermediate=False) -> torch.Tensor:
        
        inputs, labels, image_paths, mask_paths = batch
        
        outputs = self.forward(inputs)

        ''' Reshape the outputs to match the original input label's shape '''
        cropped_labels = TF.center_crop(labels, [outputs.shape[2], outputs.shape[3]])
        
        loss = self.criterion(outputs, cropped_labels)
        
         
        if write_intermediate and batch_idx % 50 == 0:
            if image_paths[0].find(stage) != -1:
                
                for j in range(0, 1):
                # for j in range(inputs.shape[0]):
                    working_images_path = os.path.join(config.OUT_DIR, "working_images", self.tag, stage)
                    epoch_path = os.path.join(working_images_path, self.tag, stage, f'epoch{self.current_epoch}')
                    os.makedirs(epoch_path, exist_ok=True)
                    original_filename = os.path.basename(image_paths[j])
                    out_filename = os.path.join(epoch_path, f'{self.current_epoch}_{batch_idx}_{original_filename}')
                
                    predicted_out_path = out_filename.replace('.png', '_predicted.png')    
                    predicted_mask_write = outputs[j].detach().cpu().numpy()
                    predicted_mask_write = np.transpose(predicted_mask_write, (1, 2, 0))
                    predicted_mask_write = np.clip(predicted_mask_write, 0, 1)
                    predicted_mask_write = (predicted_mask_write * 255)
                    cv2.imwrite(predicted_out_path, predicted_mask_write)
                    
                    actual_out_path = out_filename.replace('.png', '_actual_mask.png')    
                    actual_mask_write = cropped_labels[j, :, :, :].detach().cpu().numpy()
                    actual_mask_write = np.transpose(actual_mask_write, (1, 2, 0))
                    actual_mask_write = np.clip(actual_mask_write, 0, 1)
                    actual_mask_write = (actual_mask_write * 255)
                    cv2.imwrite(actual_out_path, actual_mask_write)
                    
                    image_out_path = out_filename.replace('.png', '_actual_image.png')    
                    actual_image_write = inputs[j].detach().cpu().numpy()
                    actual_image_write = np.transpose(actual_image_write, (1, 2, 0))
                    actual_image_write = (actual_image_write * 255)
                    cv2.imwrite(image_out_path, actual_image_write)
                    
       
        self.log(metric_name, loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=16)
        return loss
     
    def _dice_hausdorff_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor, str, str], 
        batch_idx: int, 
        metric_name : LossMetric,
        stage: str = TrainingStage.TRAIN,
        write_intermediate=False) -> torch.Tensor:
     
        ''' Validation is a bit more complex, because to do the hausdorff distance, we need to 
            work across the composition of all the slices for a particular case. So our 
            validation dataloader gives us all the data for the case at once, and we need to 
            split it into appropriate minibatches, get the results, and then recombine them
            in to a 3d object for hausdorff distance.'''
        
        inputs, actual_mask, case_dirs = batch
        
        inputs = inputs.squeeze(0)
        actual_mask = actual_mask.squeeze(0).to(dtype=torch.int)
        
        # inputs = inputs.to(device)
        # actual_mask = actual_mask.to(device).to(dtype=torch.int)
        minibatches = torch.split(inputs, split_size_or_sections=16, dim=0)
        
        predicted_mask_collections = [[] for _ in range(3)]
        slice = 0
        for i, minibatch in enumerate(minibatches):
            predicted_masks = self(minibatch)
            predicted_masks = torch.clamp(predicted_masks, 0, 1)
            
            if write_intermediate:
                case_name = os.path.basename(case_dirs[0])
                working_images_path = os.path.join(config.OUT_DIR, "_Test_Intermediate", self.tag, case_name)
                os.makedirs(working_images_path, exist_ok=True)
                for j in range(minibatch.shape[0]):
                    predicted_mask_write = predicted_masks[j].detach().cpu().numpy()
                    predicted_mask_write = np.transpose(predicted_mask_write, (1, 2, 0))
                    predicted_mask_write = (predicted_mask_write * 255)
                    filename = case_name + f'_slice_{str(slice).zfill(4)}_predicted.png'
                    cv2.imwrite(os.path.join(working_images_path, filename), predicted_mask_write)
                    
                    actual_mask_write = actual_mask[:, slice, :, :].detach().cpu().numpy()
                    actual_mask_write = np.transpose(actual_mask_write, (1, 2, 0))
                    actual_mask_write = (actual_mask_write * 255)
                    filename = case_name + f'_slice_{str(slice).zfill(4)}_actual_mask.png'
                    cv2.imwrite(os.path.join(working_images_path, filename), actual_mask_write)
                    
                    actual_image_write = minibatch[j].detach().cpu().numpy()
                    actual_image_write = np.transpose(actual_image_write, (1, 2, 0))
                    actual_image_write = (actual_image_write * 255)
                    filename = case_name + f'_slice_{str(slice).zfill(4)}_actual_image.png'
                    cv2.imwrite(os.path.join(working_images_path, filename), actual_image_write)
                    slice += 1
          
            predicted_masks = TF.center_crop(predicted_masks, [inputs.shape[2], inputs.shape[3]])             
                    
            # Split the predicted masks into separate masks for each organ
            predicted_masks_split = predicted_masks.permute(1, 0, 2, 3)
           
            # split on the channels, and make it three separate predictions, because
            # that is how we will ultimately compare them as well. 
            for i in range(3):
                predicted_mask_collections[i].append(predicted_masks_split[i])

        
        # Concatenate all the batches into one larger batch for each mask
        final_predicted_masks = [torch.cat(mask_collection, dim=0) for mask_collection in predicted_mask_collections]

        # Split the actual mask into separate masks for each organ
        actual_masks_split = torch.split(actual_mask, split_size_or_sections=1, dim=0)
        actual_masks_split = [torch.squeeze(mask, dim=0) for mask in actual_masks_split]

        mse_loss_accumulator = 0
        dice_accumulator = 0
        hausdorff_accumulator = 0

        for predicted_mask, actual_mask in zip(final_predicted_masks, actual_masks_split):
            # predicted_mask = actual_mask.to(dtype=torch.float32)   
            # Temp to that the losses are calcualted correctly when it's 100%
            # predicted_mask = actual_mask.to(dtype=torch.float32)
            mse_loss_accumulator += self.criterion(predicted_mask, actual_mask).item()
        
            #  Dice is the score, so we need 1 - dice to get loss
            # dice_loss_accumulator += 1 - self.dice(predicted_mask, actual_mask).item() 
            dice_similarity_score = self.dice(predicted_mask, actual_mask).item()
            dice_accumulator += dice_similarity_score
            
            hausdorff_similarity_distance_score = self._hausdorff(predicted_mask, actual_mask)
            hausdorff_accumulator += hausdorff_similarity_distance_score
            
        average_mse_loss = mse_loss_accumulator / 3
        average_dice_similarity = dice_accumulator / 3
        average_inverse_distance = hausdorff_accumulator / 3
        mixed_score = (0.6 * average_inverse_distance) + (0.4 * average_dice_similarity)

        values = {
            LossMetric.MSE_LOSS: average_mse_loss, 
            LossMetric.DICE_LOSS: 1 - average_dice_similarity,
            LossMetric.HAUSDORFF_LOSS: 1 - average_inverse_distance,
            LossMetric.MIXED_LOSS: mixed_score}
        self.log_dict(values,  on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return 1 - average_inverse_distance

    def _hausdorff(self, predicted_mask_3d, actual_mask_3d):
        # normalize loss_score by dividing by width * height
        # the depth for each slice is 1, so we don't need to divide by that.
        
        # Measures the degree of similarity between two Geometrys using the Hausdorff distance metric. 
        # The measure is normalized to lie in the range [0, 1]. Higher measures indicate a great degree of similarity.
        # The measure is computed by computing the Hausdorff distance between the input geometries, and then normalizing 
        # this by dividing it by the diagonal distance across the envelope of the combined geometries.
        
        all_predicted_points = np.empty((0, 3))
        all_actual_points = np.empty((0, 3))
        for i in range(0, predicted_mask_3d.shape[0]):
            
            
            predicted = predicted_mask_3d[i, :, :] # .unsqueeze(0).unsqueeze(0)
            
            np_predicted = predicted.detach().cpu().numpy()
            np_predicted = np.clip(np_predicted, 0, 1)
            np_predicted = (np_predicted * 255).astype(np.uint8)
            edges = cv2.Canny(np_predicted,100,200)
            y_coords, x_coords = np.where(edges > 0)
            z_coords = np.full_like(x_coords, i)
            predicted_points = np.column_stack((x_coords, y_coords, z_coords))
            all_predicted_points = np.concatenate((all_predicted_points, predicted_points))
         
            # For debugging   
            # out_file = os.path.join(canny_out, f"canny_predicted{i}.png")
            # os.makedirs(os.path.dirname(out_file), exist_ok=True)
            # cv2.imwrite(out_file, edges)
            
            actual = actual_mask_3d[i, :, :] # .unsqueeze(0).unsqueeze(0)
            np_actual = actual.detach().cpu().numpy()
            np_actual = np.clip(np_actual, 0, 1)
            np_actual = (np_actual * 255).astype(np.uint8)
            edges = cv2.Canny(np_actual,100,200)
            y_coords, x_coords = np.where(edges > 0)
            z_coords = np.full_like(x_coords, i)
            actual_points = np.column_stack((x_coords, y_coords, z_coords))
            all_actual_points = np.concatenate((all_actual_points, actual_points))
            
            # For debugging
            # out_file = os.path.join(canny_out, f"canny_actual{i}.png")
            # cv2.imwrite(out_file, edges)
            
        dist = self.hausdorff_distance_3d(all_predicted_points, all_actual_points)
        
        if dist == np.inf:
            return 0
        diagonal = np.sqrt(actual.shape[0]**2 + actual.shape[1]**2)
        normalized_score = dist / diagonal
        
        return 1 - normalized_score  

    def hausdorff_distance_3d(self, setA, setB):
        """
        Compute the Hausdorff Distance between two 3D point sets.

        :param setA: numpy.ndarray of shape (n_points, 3)
        :param setB: numpy.ndarray of shape (m_points, 3)
        :return: Hausdorff distance as a float
        """
        # Compute pairwise distance matrix between points in set A and set B
        distances = cdist(setA, setB, 'euclidean')

        if (len(setA) == 0 or len(setB) == 0):
            return np.inf

        # Find the single closest point in set B for each point in set A, and vice versa
        min_dist_to_B = distances.min(axis=1)
        min_dist_to_A = distances.min(axis=0)

        # The Hausdorff distance is the max of these minimum distances
        hausdorff_dist = max(np.max(min_dist_to_B), np.max(min_dist_to_A))
        return hausdorff_dist
