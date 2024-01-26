# Standard library imports
import math
import os
from collections import namedtuple

# Related third party imports
import cv2
import numpy as np

SegmentationCollection = namedtuple('SegmentationCollection', 'Color24Bit Color48Bit StomachMask, LargeBowelMask, SmallBowelMask')

def get_single_channel_grayscale_image(img_path: str) -> np.array:
     # Load the 16-bit grayscale image using OpenCV
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if img.dtype != 'uint16':
        print("The image is not in 16-bit grayscale format.")
        return

    # Ensure the image is 16-bit grayscale
    if len(img.shape) != 2:
        print("The image is not a single-channel grayscale image.")
        return

    # max_val = img.max()
    # img = img.astype(dtype='float') / max_val
    # img = img * 2**16
    # img = cv2.merge([img, img, img])

    return img
  

def convert_original_image_to_24bit_48bit(img_path: str, min: float = -1, max: float = -1) -> np.array:
    
    if min != -1 and max != -1:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # Load the 16-bit grayscale image using OpenCV
    else:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if min == -1:
            min = img.min()
        if max == -1:
            max = img.max()

    if img.dtype != 'uint16':
        print("The image is not in 16-bit grayscale format.")
        return

    # Ensure the image is 16-bit grayscale
    if len(img.shape) != 2:
        print("The image is not a single-channel grayscale image.")
        return

    # Scale the array to 8bit
    scaled_array = (img - min) * (255.0 / max - min)
    scaled_array.astype(np.uint8, copy=False)

    rgb_array_24bit = cv2.merge([scaled_array, scaled_array, scaled_array])
    rbg_array_48bit = cv2.merge([img, img, img])

    return img.shape, rgb_array_24bit, rbg_array_48bit

def generate_segmented_image_24bit_48bit(
        shape: (int,int), 
        organ_dict: dict[str, list[tuple[int, int]]]) -> SegmentationCollection:

    # Initialize an empty array for the 16-bit RGB image
    color_map_16 = np.array([
            [0,     0,     0    ],       # 0 - Black
            [0,     0,     255  ],       # 1 - Red - Stomach
            [0,     255,   0    ],       # 2 - Green - Large_Bowel
            [255,   0,     0    ]        # 3 - Blue - Small_Bowel
        ], dtype=np.uint8)
    
    bgr_array_24 = generate_segmented_image(shape, organ_dict, color_map_16)
    
    color_map_48 = np.array([
        [0,     0,     0    ],       # 0 - Black
        [0,     0,     65535],       # 1 - Red - Stomach
        [0,     65535, 0    ],       # 2 - Green - Large_Bowel
        [65535, 0,     0    ]        # 3 - Blue - Small_Bowel
    ])

    bgr_array_48 = generate_segmented_image(shape, organ_dict, color_map_48)
    
    mask_map = np.array([
        0,       # 0 - Black
        255,     # 1 - White - Stomach
        255,     # 2 - White - Large_Bowel
        255      # 3 - White - Small_Bowel
    ])

    single_organ_dict = {}
    single_organ_dict["stomach"] = organ_dict["stomach"]      
    stomach_mask_array = generate_segmented_image(shape, single_organ_dict, mask_map)

    single_organ_dict = {}
    single_organ_dict["large_bowel"] = organ_dict["large_bowel"]      
    large_bowel_mask_array = generate_segmented_image(shape, single_organ_dict, mask_map)

    single_organ_dict = {}
    single_organ_dict["small_bowel"] = organ_dict["small_bowel"]      
    small_bowel_mask_array = generate_segmented_image(shape, single_organ_dict, mask_map)

    return SegmentationCollection(bgr_array_24, bgr_array_48, stomach_mask_array, large_bowel_mask_array, small_bowel_mask_array)


def generate_segmented_image(
        img_shape: tuple[int, int],
        organ_dict: dict[str, list[tuple[int, int]]],
        in_color_map: np.array) -> np.array:
    """
    Generate the segmented image.
    """

    organ_index = {
    "stomach": 1,
    "large_bowel": 2,
    "small_bowel": 3
    }

    image_array = np.zeros(img_shape[0] * img_shape[1], dtype=np.uint8)

    # Assuming organ_dict is defined and contains organ segmentation data
    for organ_type, segmentations in organ_dict.items():
        organ_color_index = organ_index[organ_type]
        for offset, length in segmentations:
            image_array[offset:offset+length] = organ_color_index

    # Reshape the array into an image shape
    image_array = image_array.reshape((img_shape[0], img_shape[1]))

    out_array = in_color_map[image_array]
            
    return out_array


def overlay_original_image_24bit_48bit(segmented_24bit: np.array, 
                                       segmented_48bit: np.array, 
                                       img_path: str) -> np.array:

    if not os.path.exists(img_path):
        print(f"Base path {img_path} does not exist.")
        return

    _, original_24bit, original_48bit = convert_original_image_to_24bit_48bit(img_path)

    if (original_24bit.shape != original_48bit.shape):
        print("The original image and the colored image are not the same shape.")
        print(img_path)
        return

    # Blend the grayscale image and the color image
    # OpenCV blend: dst = src1*alpha + src2*beta + gamma
    # To match Image.blend behavior: alpha = 0.3, beta = 1 - alpha, gamma = 0
    alpha = .9
    beta = 1 - alpha
    gamma = 0.0
    overlay_image_24bit = cv2.addWeighted(original_24bit, alpha, segmented_24bit, beta, gamma)
    overlay_image_48bit = cv2.addWeighted(original_48bit, alpha, segmented_48bit, beta, gamma)

    return overlay_image_24bit, overlay_image_48bit

def normalize_pixel_size(image: np.ndarray, 
                         source_pixel_width : float, 
                         source_pixel_height: float, 
                         source_image_width: int,
                         source_image_height: int) -> np.ndarray:
    '''
        The pixels are usually 1.55 or 1.66 mm in width and height.
        This function normalizes the pixel size to 1 mm.
    '''

    new_image_width = math.ceil(source_pixel_width * source_image_width)
    new_image_height = math.ceil(source_pixel_height * source_image_height)
    
    dest_image = cv2.resize(image, (new_image_width, new_image_height), interpolation=cv2.INTER_LINEAR_EXACT)
    return dest_image


def reshape_image(image: np.ndarray, target_shape_width: int, target_shape_height: int) -> np.ndarray:
    """
    Reshape the image to the target shape
    """
    # numpy uses (height, width) for shape.
    image_shape_height, image_shape_width = image.shape[0], image.shape[1]

    if target_shape_width < image_shape_width or target_shape_height < image_shape_height:
        print(f"Target shape {target_shape_width}x{target_shape_height} is smaller than image shape {image_shape_width}x{image_shape_height}.")
        return image

    l = int(target_shape_width - image_shape_width) // 2
    r = int(target_shape_width - image_shape_width - l)
    
    t = int(target_shape_height - image_shape_height) // 2
    b = int(target_shape_height - image_shape_height - t)

    if len(image.shape) == 2:
        padded_image = cv2.copyMakeBorder(image, t, b, l, r, cv2.BORDER_CONSTANT, value=0)
    else:
        padded_image = cv2.copyMakeBorder(image, t, b, l, r, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return padded_image


