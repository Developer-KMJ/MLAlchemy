
import os
import cv2
import numpy as np


def get_image_cache_from_directory(path : str) -> np.ndarray:
    
    imageFiles = []
    for imgFile in sorted(os.listdir(path)):
        if imgFile.endswith('.png'):
            imageFiles.append(imgFile)

    img = cv2.imread(os.path.join(path, imageFiles[0]))
    height = img.shape[0]
    width = img.shape[1]
    sliceCount = len(imageFiles)

    image_cache = []
    for j_slice in range(sliceCount):
        full_path = os.path.join(path, imageFiles[j_slice])
        img = cv2.imread(full_path)
        image_cache.append(img)

    final_cache = np.array(image_cache)
    return final_cache