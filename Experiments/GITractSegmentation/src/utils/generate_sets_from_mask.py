import cv2
import numpy as np

def extract_edge_points(image_mask):
    # Threshold the image
    _, thresh_mask = cv2.threshold(image_mask, 128, 255, cv2.THRESH_BINARY)

    # Apply Canny edge detector
    edges = cv2.Canny(thresh_mask, 100, 200)

    # Find coordinates of edge points
    # np.argwhere will give you the (row, col) coordinates of non-zero points
    edge_points = np.argwhere(edges > 0)

    # Convert to (x, y) format instead of (row, col)
    edge_points = np.flip(edge_points, axis=1)

    return edge_points

