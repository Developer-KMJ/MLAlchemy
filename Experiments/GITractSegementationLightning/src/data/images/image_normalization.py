
import math
import os
import numpy as np
from data.images.image_conversion import get_single_channel_grayscale_image

def get_image_normalization_values_records(record_data: dict[str, str]) -> (float, float, float, float, float, float, float, float):
    file_list = []
    for value in record_data.values():
        file_list.append(value.scan_file)
    
    return __get_image_normalization_values_common(file_list)    

def get_image_normalization_values_directory(data_directory: str) -> (float, float, float, float, float, float, float):
    
    file_list = []
    for root, _, files in os.walk(data_directory):
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(root, file)
                file_list.append(img_path)
               
                
    return __get_image_normalization_values_common(file_list)

def __get_image_normalization_values_common(data_directory: [str]) -> (float, float, float, float, float, float, float):
    """
    Get the min and max values for the images in the data directory.
    """
    min_val = 0
    max_val = 0

    counts = np.zeros(65536, dtype=int) 
    for img_path in data_directory:
        
        img = get_single_channel_grayscale_image(img_path)
        
        # keep track of how many of each pixel value there are
        # in the image.
        newcounts = np.bincount(img.flatten(), minlength=65536)
        counts += newcounts


    sum_elements = np.sum([index * value for index, value in enumerate(counts)])
    num_elements = np.sum(counts)
              
    for index, value in enumerate(counts):
        if value > 0:
            min_val = index
            break
        
    for index, value in enumerate(counts[::-1]):
        if value > 0:
            max_val = 65535 - index
            break
    
    mean = sum_elements/num_elements
    
    original_variance = np.sum([(index - mean)**2 * value for index, value in enumerate(counts)])/num_elements
    original_stddev = math.sqrt(original_variance)

    # The long way
    # sum_elements_scaled = np.sum([(index/max_val) * value for index, value in enumerate(counts)])    
    # mean_scaled = sum_elements_scaled/num_elements
    # scaled_variance = np.sum([((index/max_val) - mean_scaled)**2 * value for index, value in enumerate(counts)])/num_elements
    # scaled_stddev = math.sqrt(scaled_variance)

    # Applying a conversion factor.
    mean_scaled = (1/(max_val - min_val)) * mean
    scaled_variance = (1/(max_val - min_val))**2 * original_variance
    scaled_stddev = math.sqrt(scaled_variance)
    
    return min_val, max_val, mean, original_stddev, original_variance, mean_scaled, scaled_stddev, scaled_variance

if __name__ == '__main__':
    # Test the functions
    data_directory = '/home/kevin/Documents/MRIProject-Working/data/processed/temp/train'
    min_val, max_val, mean, original_stddev, original_variance, scaled_mean, scaled_stddev, scaled_variance = get_image_normalization_values_directory(data_directory)
    print(f"Min: {min_val}, Max: {max_val}")
    print(f"Mean: {mean}, Original StdDev: {original_stddev}, Original Variance: {original_variance}")
    print(f"Scaled Mean:{scaled_mean} Scaled StdDev: {scaled_stddev}, Scaled Variance: {scaled_variance}")
    print("Done.")
    