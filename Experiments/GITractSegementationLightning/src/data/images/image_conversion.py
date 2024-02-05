import cv2
import numpy as np
import os

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

