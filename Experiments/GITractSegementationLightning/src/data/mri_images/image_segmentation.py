
from collections import namedtuple
import numpy as np


SegmentationCollection = namedtuple('SegmentationCollection', 'Color24Bit Color48Bit StomachMask, LargeBowelMask, SmallBowelMask')

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

