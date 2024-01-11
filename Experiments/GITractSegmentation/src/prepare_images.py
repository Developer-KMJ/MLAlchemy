
import argparse
import os
import cv2
from utils.csv_data_parser import RecordData, parse_csv_into_record_data
from utils.raw_image_processor import convert_original_image_to_24bit_48bit, generate_segmented_image_24bit_48bit, get_single_channel_grayscale_image, normalize_pixel_size, overlay_original_image_24bit_48bit, reshape_image
import torch
import numpy as np

def main():
    args = parse_args()

    train_dir = args.base_path
    train_csv_path = args.csv_path
    out_path = args.out_path

    # get_stats_for_dir(train_dir)
    # return

    data_records = parse_csv_into_record_data(train_dir, train_csv_path)
    if (data_records is None or len(data_records) == 0):
        print(f"Could not find segmentation data files in {train_dir}")
        return

    # every image needs to be normalized to 1 pixel = 1 mm and then reshaped to the maximum
    # so that all the data is the same physical size and image size. 
    max_width =  max(map(lambda x: x[1].pixel_width * x[1].image_width, data_records.items()))
    max_height =  max(map(lambda x: x[1].pixel_width * x[1].image_height, data_records.items()))

    if args.walk_path is False:
        if (args.case is None or args.day is None or args.slice_number is None):
            print("Case, day, and slice number must be specified.")
            return

        key = f'case{args.case}_day{args.day}_slice_{str(args.slice_number).zfill(4)}'
        reduced_records = {}
        reduced_records[key] = data_records[key]
        if len(reduced_records) == 0:
            print(f"Could not find segmentation data for {key}")
            return

        data_records = reduced_records

    # every image needs to be normalized to 1 pixel = 1 mm and then reshaped to the maximum
    # so that all the data is the same physical size and image size. 
    max_width =  max(map(lambda x: x[1].pixel_width * x[1].image_width, data_records.items()))
    max_height =  max(map(lambda x: x[1].pixel_width * x[1].image_height, data_records.items()))

    if (max_width > 572 or max_height > 572):
        print("The maximum width and height of the images is greater than 572 pixels, so resizing needs to be handled differently")
        return

    max_height = 572
    max_width = 572

    for key in sorted(data_records.keys()):  
      if data_records[key].has_organs is True:  
        generate_images_and_masks(data_records[key], out_path, max_width, max_height)

        

    

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="test_main.py",
        description="Convert csv file to segmented image.")
    
    parser.add_argument("--csv_path", type=str, help="Path to segmentation csv file", required=True)
    parser.add_argument("--base_path", type=str, help="Path to the training data.", required=True)
    parser.add_argument("--out_path", type=str, help="Path to save segmented image.", required=True)

    parser.add_argument("--walk_path", type=bool, help="Run generation of every item in the path.", default=False)

    parser.add_argument("--case", type=int, help="Case ID of segmentation data in csv file.", required=False)
    parser.add_argument("--day", type=int, help="Day of segmentation data in csv file.", required=False)
    parser.add_argument("--slice_number", type=int, help="Slice of segmentation data in csv file.", required=False)

    args = parser.parse_args()
    return args

def generate_images(data_record: RecordData, out_path: str, target_shape_width: int, target_shape_height: int):
    """
    Generate the images from the data records
    """

    if (out_path is None):
        raise ValueError("out_path cannot be None")
    
    save_path = os.path.join(out_path, data_record.label)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    shape, original_color_image_24bit, original_color_image_48bit = convert_original_image_to_24bit_48bit(data_record.scan_file)
    cv2.imwrite(os.path.join(save_path, 'normal_size_image_24.tiff'), original_color_image_24bit)
    
    original_color_image_24bit = normalize_pixel_size(original_color_image_24bit, data_record.pixel_width, data_record.pixel_height, data_record.image_width, data_record.image_height)
    original_color_image_24bit = reshape_image(original_color_image_24bit, target_shape_width, target_shape_height)
    cv2.imwrite(os.path.join(save_path, 'original_image_24.tiff'), original_color_image_24bit)

    '''
    original_color_image_48bit = normalize_pixel_size(original_color_image_48bit, data_record.pixel_width, data_record.pixel_height, data_record.image_width, data_record.image_height)
    original_color_image_48bit = reshape_image(original_color_image_48bit, target_shape)
    cv2.imwrite(os.path.join(save_path, 'original_image_48.tiff'), original_color_image_48bit)
    '''

    segmentCollection = generate_segmented_image_24bit_48bit(shape, data_record.organ_list)

    '''
    img = normalize_pixel_size(segmentCollection.Color24Bit, data_record.pixel_width, data_record.pixel_height, data_record.image_width, data_record.image_height)
    img = reshape_image(img, target_shape)
    cv2.imwrite(os.path.join(save_path, 'segmented_image_24.tiff'), img)

    img = normalize_pixel_size(segmentCollection.Color48Bit, data_record.pixel_width, data_record.pixel_height, data_record.image_width, data_record.image_height)
    img = reshape_image(img, target_shape)
    cv2.imwrite(os.path.join(save_path, 'segmented_image_48.tiff'), img)
    ''' 

    img = normalize_pixel_size(segmentCollection.StomachMask, data_record.pixel_width, data_record.pixel_height, data_record.image_width, data_record.image_height)
    img = reshape_image(img, target_shape_width, target_shape_height)
    cv2.imwrite(os.path.join(save_path, 'stomach_mask.tiff'), img)

    img = normalize_pixel_size(segmentCollection.LargeBowelMask, data_record.pixel_width, data_record.pixel_height, data_record.image_width, data_record.image_height)
    img = reshape_image(img, target_shape_width, target_shape_height)
    cv2.imwrite(os.path.join(save_path, 'large_bowel_mask.tiff'), img)

    img = normalize_pixel_size(segmentCollection.SmallBowelMask, data_record.pixel_width, data_record.pixel_height, data_record.image_width, data_record.image_height)
    img = reshape_image(img, target_shape_width, target_shape_height)
    cv2.imwrite(os.path.join(save_path, 'small_bowel_mask.tiff'), img)
    
    '''
    overlay_image_24bit, overlay_image_48bit = overlay_original_image_24bit_48bit(
            segmentCollection.Color24Bit, 
            segmentCollection.Color48Bit, 
            data_record.scan_file)
        
    img = normalize_pixel_size(overlay_image_24bit, data_record.pixel_width, data_record.pixel_height, data_record.image_width, data_record.image_height)
    img = reshape_image(img, target_shape_width, target_shape_height)
    cv2.imwrite(os.path.join(save_path, 'overlay_image_24.tiff'), img)

    img = normalize_pixel_size(overlay_image_48bit, data_record.pixel_width, data_record.pixel_height, data_record.image_width, data_record.image_height)
    img = reshape_image(img, target_shape)
    cv2.imwrite(os.path.join(save_path, 'overlay_image_48.tiff'), img)
    '''

def generate_images_and_masks(data_record: RecordData, out_path: str, target_shape_width: int, target_shape_height: int):

    if (out_path is None):
        raise ValueError("out_path cannot be None")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    img_path = os.path.join(out_path, 'images')
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    mask_path = os.path.join(out_path, 'masks')
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    shape, original_color_image_24bit, original_color_image_48bit = convert_original_image_to_24bit_48bit(data_record.scan_file)
    
    original_color_image_24bit = normalize_pixel_size(original_color_image_24bit, data_record.pixel_width, data_record.pixel_height, data_record.image_width, data_record.image_height)
    original_color_image_24bit = reshape_image(original_color_image_24bit, target_shape_width, target_shape_height)
    cv2.imwrite(os.path.join(img_path, f'{data_record.label}.png'), original_color_image_24bit)

    segmentCollection = generate_segmented_image_24bit_48bit(shape, data_record.organ_list)
    
    img = normalize_pixel_size(segmentCollection.Color24Bit, data_record.pixel_width, data_record.pixel_height, data_record.image_width, data_record.image_height)
    img = reshape_image(img, target_shape_width, target_shape_height)
    cv2.imwrite(os.path.join(mask_path, f'{data_record.label}.png'), img)

def get_stats_for_dir(path):
 
    full_aggregate = (0, 0, 0)
    for file in os.listdir(path):
        if file.endswith(".png"):
            img = cv2.imread(os.path.join(path, file), cv2.IMREAD_UNCHANGED)
            img_channel = img[:,:,0]

            img_channel = img_channel/255.0
            instance_mean = np.mean(img_channel)
           
            full_aggregate = update(full_aggregate, instance_mean)

      #  generate_images_and_masks(data_records[key], out_path, max_width, max_height)

    mean, variance, sample_variance = finalize(full_aggregate)
   
    print(f'Mean: {mean}')
    print(f'Variance: {variance}')

# Welford's algorithm for computing a running mean/variance
# For a new value new_value, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existing_aggregate, new_value):
    (count, mean, M2) = existing_aggregate
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return (count, mean, M2)

# Retrieve the mean, variance and sample variance from an aggregate
def finalize(existing_aggregate):
    (count, mean, M2) = existing_aggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sample_variance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sample_variance)
    


if __name__ == "__main__":
    main()
    

