import cv2
import numpy as np
import torch
import torch.nn as nn


#PyTorch
class HausdorffLoss(nn.Module):
    def __init__(self):
        super(HausdorffLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Assuming batch_pred and batch_target are batches of images
        # Convert images to sets of points (could be edge detection or other feature extraction)
        # For simplicity, let's assume function images_to_points() does this
        batch_pred_points, batch_target_points = self.__images_to_points(inputs), self.__images_to_points(targets)

        # Compute Hausdorff Distance for each pair in the batch
        losses = []
        for pred, target in zip(batch_pred_points, batch_target_points):
            loss = self.__hausdorff_distance(pred, target)
            losses.append(loss)

        # Aggregate the losses
        total_loss = torch.mean(torch.stack(losses))
        return total_loss    

    def __hausdorff_distance(setA, setB):
        # This function calculates the Hausdorff Distance between two sets of points
        # Assuming setA and setB are tensors of points (N, 2) and (M, 2)

        # Expand dims to (N, 1, 2) and (1, M, 2) to compute pairwise distance
        expanded_A = setA.unsqueeze(1)
        expanded_B = setB.unsqueeze(0)

        # Compute pairwise Euclidean distance
        distances = torch.cdist(expanded_A, expanded_B, p=2)  # (N, M)

        # Hausdorff Distance: max(min distance from a point in A to B, and vice versa)
        hdist_A_to_B = distances.min(dim=1)[0].max()  # max of min distances from A to B
        hdist_B_to_A = distances.min(dim=0)[0].max()  # max of min distances from B to A

        return max(hdist_A_to_B, hdist_B_to_A)

    def __images_to_points(images, mid_value = 128, max_value = 255):
        batch_edge_points = []

        #TODO: If the images are 4d instead of 3, that means we have a batch of 3d images so we need to flatten
        # it to 3d of len x width x (batch X channel)

        for image in images:
            # Threshold the image
            _, thresh_mask = cv2.threshold(image, mid_value, max_value, cv2.THRESH_BINARY)

            # Apply Canny edge detector
            edges = cv2.Canny(thresh_mask, 100, 200)

            # Find coordinates of edge points
            # np.argwhere will give you the (row, col) coordinates of non-zero points
            edge_points = np.argwhere(edges > 0)

            # Convert to (x, y) format instead of (row, col)
            edge_points = np.flip(edge_points, axis=1)

            batch_edge_points.append(edge_points)

        return batch_edge_points
        

