''' 
    Creates the dataset for our first stage which will use yolo to determine:
    1) Does the image contain the organs of interest in general?
    2) What size is the main torso that is being scanned, so that we can process down to that size. 
'''


import argparse
import os
import cv2
from utils.csv_data_parser import RecordData, generate_train_csv_file, parse_csv_into_record_data
from utils.raw_image_processor import convert_original_image_to_24bit_48bit, generate_segmented_image_24bit_48bit, get_single_channel_grayscale_image, normalize_pixel_size, overlay_original_image_24bit_48bit, reshape_image

def main():
    args = parse_args()

    train_dir = args.base_path
    train_csv_path = args.csv_path
    out_path = args.out_path

    data_records = parse_csv_into_record_data(train_dir, train_csv_path)
    if (data_records is None or len(data_records) == 0):
        print(f"Could not find segmentation data files in {train_dir}")
        return

    if args.regenerate_csv is True:
        if os.path.exists(os.path.join(out_path, 'train.csv')):
            os.rename(os.path.join(out_path, 'train.csv'), os.path.join(out_path, 'train.csv.bak'))
            
        new_csv = os.path.join(out_path, 'train.csv')
        generate_train_csv_file(data_records, new_csv)
    
    # every image needs to be normalized to 1 pixel = 1 mm and then reshaped to the maximum
    # so that all the data is the same physical size and image size. 
    max_width =  max(map(lambda x: x[1].pixel_width * x[1].image_width, data_records.items()))
    max_height =  max(map(lambda x: x[1].pixel_width * x[1].image_height, data_records.items()))

    # every image needs to be normalized to 1 pixel = 1 mm and then reshaped to the maximum
    # so that all the data is the same physical size and image size. 
    max_width =  max(map(lambda x: x[1].pixel_width * x[1].image_width, data_records.items()))
    max_height =  max(map(lambda x: x[1].pixel_width * x[1].image_height, data_records.items()))

    for key in sorted(data_records.keys()):
        generate_images(data_records[key], out_path, max_width, max_height)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="prepare_ioi.py",
        description="Split the datasets into images of interest and those that are not.")
    
    parser.add_argument("--csv_path", type=str, help="Path to segmentation csv file", required=True)
    parser.add_argument("--base_path", type=str, help="Path to the training data.", required=True)
    parser.add_argument("--out_path", type=str, help="Path to save segmented image.", required=True)

    parser.add_argument("--regenerate_csv", type=bool, help="Regenerate the train csv file.", default=False)

    args = parser.parse_args()
    return args

def generate_images(data_record: RecordData, out_path: str, target_shape_width: int, target_shape_height: int):
    """
    Generate the images from the data records
    """

    if (out_path is None):
        raise ValueError("out_path cannot be None")
    
    ''' Make sure that all of these paths exist prior to processing.'''
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    no_organs_path = os.path.join(out_path, 'no_organs')
    if not os.path.exists(no_organs_path):
        os.makedirs(no_organs_path)

    organs_path = os.path.join(out_path, 'organs')
    if not os.path.exists(organs_path):
        os.makedirs(organs_path)


    if data_record.has_organs is False:
        out_path = no_organs_path
    else:
        out_path = organs_path

    # original_image_16bit = get_single_channel_grayscale_image(data_record.scan_file)
    _, original_8, original_16bit = convert_original_image_to_24bit_48bit(data_record.scan_file)   
    
    original_8 = normalize_pixel_size(original_8, data_record.pixel_width, data_record.pixel_height, data_record.image_width, data_record.image_height)
    original_8 = reshape_image(original_8, target_shape_width, target_shape_height)
    cv2.imwrite(os.path.join(out_path, data_record.label + ".tiff"), original_8)

    original_16bit = normalize_pixel_size(original_16bit, data_record.pixel_width, data_record.pixel_height, data_record.image_width, data_record.image_height)
    original_16bit = reshape_image(original_16bit, target_shape_width, target_shape_height)
    cv2.imwrite(os.path.join(out_path, data_record.label + ".tiff"), original_16bit)

    
if __name__ == "__main__":
    main()
    

