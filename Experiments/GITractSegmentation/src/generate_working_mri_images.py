
import argparse
import math
import os
import shutil
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

    axial_path = os.path.join(out_path, f'Axial')
    axial_image_path = os.path.join(axial_path, 'images')
    axial_mask_path = os.path.join(axial_path, 'masks')

    coronal_path = os.path.join(args.out_path, f'Coronal')
    coronal_image_path = os.path.join(coronal_path, 'images')
    coronal_mask_path = os.path.join(coronal_path, 'masks')

    sagittal_path = os.path.join(args.out_path, f'Sagittal')
    sagittal_image_path = os.path.join(sagittal_path, 'images')
    sagittal_mask_path = os.path.join(sagittal_path, 'masks')

    # Walk through data by Case Number, as we will need to generate images using all the 
    # slices in that directory.
    case_numbers = sorted(set(record.case for record in data_records.values()))
    for case_number in case_numbers:
        days = sorted(set({record.day for record in data_records.values() if record.case == case_number}))
        for day in days:

            case_axial_image_path = os.path.join(axial_image_path, f'case{case_number}_day{day}')
            case_axial_mask_path = os.path.join(axial_mask_path, f'case{case_number}_day{day}')
            case_coronal_image_path = os.path.join(coronal_image_path, f'case{case_number}_day{day}')
            case_coronal_mask_path = os.path.join(coronal_mask_path, f'case{case_number}_day{day}')
            case_sagittal_image_path = os.path.join(sagittal_image_path, f'case{case_number}_day{day}')           
            case_sagittal_mask_path = os.path.join(sagittal_mask_path, f'case{case_number}_day{day}')           

            # case_records = [record for record in data_records.values() if record.case == case_number and record.day == day]
            # sorted_records = sorted(case_records, key=lambda x: (int(x.case), int(x.day), int(x.slice)))
            # if len(sorted_records) != 144:
            #     print(f'Case {case_number} day {day} does not have 144 slices. Skipping.')

            generate_axial_mri(train_dir, case_axial_image_path, case_axial_mask_path, case_number, day,
                        data_records, max_width, max_height)

            # Generate the coronal MRI images for the specific case and day,
            # using the axial images we just generated.
            generate_coronal_from_axial_mri(case_axial_image_path, case_coronal_image_path, case_number, day)
            generate_coronal_from_axial_mri(case_axial_mask_path, case_coronal_mask_path, case_number, day)
                
            # Generate the sagittal MRI images for the specific case and day
            generate_sagittal_from_axial_mri(case_axial_image_path, case_sagittal_image_path, case_number, day)
            generate_sagittal_from_axial_mri(case_axial_mask_path, case_sagittal_mask_path, case_number, day)

    StageDirectories(out_path, out_path + '_staged')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="test_main.py",
        description="Convert csv file to segmented image.")
    
    parser.add_argument("--csv_path", type=str, help="Path to segmentation csv file", required=True)
    parser.add_argument("--base_path", type=str, help="Path to the training data.", required=True)
    parser.add_argument("--out_path", type=str, help="Path to save segmented image.", required=True)

    args = parser.parse_args()
    return args

def generate_axial_mri(axial_slices_path : str, 
                       output_path_image: str, 
                       output_path_mask: str, 
                       case: str, 
                       day: str,
                       data_records : [RecordData], target_shape_width: int, target_shape_height: int):
   
    
    case_day_path = os.path.join(axial_slices_path, f'case{case}', f'case{case}_day{day}', 'scans')
    img_max, img_min, _, _ = _get_normalization_values(case_day_path)   

    if not os.path.exists(output_path_image):
        os.makedirs(output_path_image)

    if not os.path.exists(output_path_mask):
        os.makedirs(output_path_mask)

    case_records = [record for record in data_records.values() if record.case == case and record.day == day]
    sorted_records = sorted(case_records, key=lambda x: (int(x.case), int(x.day), int(x.slice)))
    for i, slice_record in enumerate(sorted_records):
        shape, original_color_image_24bit, _ = convert_original_image_to_24bit_48bit(slice_record.scan_file, img_min, img_max)
        original_color_image_24bit = normalize_pixel_size(original_color_image_24bit, slice_record.pixel_width, slice_record.pixel_height, slice_record.image_width, slice_record.image_height)
        original_color_image_24bit = reshape_image(original_color_image_24bit, target_shape_width, target_shape_height)

        name = f'case{case}_day{day}_slice_{str(i).zfill(4)}.png'
        cv2.imwrite(os.path.join(output_path_image, name), original_color_image_24bit)


        segmentCollection = generate_segmented_image_24bit_48bit(shape, slice_record.organ_list)        
        img = normalize_pixel_size(segmentCollection.Color24Bit, slice_record.pixel_width, slice_record.pixel_height, slice_record.image_width, slice_record.image_height)
        img = reshape_image(img, target_shape_width, target_shape_height)
        
        name = f'case{case}_day{day}_slice_{str(i).zfill(4)}.png'
        cv2.imwrite(os.path.join(output_path_mask, name), img)

    # If we have less than 144 slices, we need to pad the images with black images
    if len(sorted_records) < 144:
        print(f'Case {case} day {day} does not have 144 slices. Padding with black images.')
        blank = np.zeros((int(target_shape_height), int(target_shape_width), 3), dtype=np.uint8)
        for i in range(len(sorted_records), 144):
            name = f'case{case}_day{day}_slice_{str(i).zfill(4)}.png'
            cv2.imwrite(os.path.join(output_path_image, name), blank)
            cv2.imwrite(os.path.join(output_path_mask, name), blank)


def generate_coronal_from_axial_mri(axial_slices_path : str, output_path: str, case: str, day: str):
    
    cache = __get_image_cache_from_directory(axial_slices_path)
    __save_sliced_images(cache, output_path, (1, 0, 2, 3), case, day, scale_height=3)

def generate_axial_from_coronal_mri(coronal_slices_path : str, output_path: str, case: str, day: str):

    cache = __get_image_cache_from_directory(coronal_slices_path)
    __save_sliced_images(cache, output_path, (1, 0, 2, 3), case, day, reduce_cache=3)

def generate_sagittal_from_axial_mri(axial_slices_path : str, output_path: str, case: str, day: str):

    cache = __get_image_cache_from_directory(axial_slices_path)
    __save_sliced_images(cache, output_path, (2, 1, 0, 3), case, day, scale_width=3)

def generate_axial_from_sagittal_mri(sagittal_slices_path : str, output_path: str, case: str, day: str):

    cache = __get_image_cache_from_directory(sagittal_slices_path)
    __save_sliced_images(cache, output_path, (2, 1, 0, 3), case, day, reduce_cache=3)


def __get_image_cache_from_directory(path : str) -> np.ndarray:
    
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

def __save_sliced_images(cache : np.ndarray, 
                   output_path: str, 
                   axis: (int, int, int, int), 
                   case: str, 
                   day: str,
                   scale_width: float = 1, 
                   scale_height: float = 1,
                   reduce_cache: int = 1):
    
    os.makedirs(output_path, exist_ok=True)

    temp_cache = cache.transpose(axis)

    if reduce_cache > 1:
        temp_cache = temp_cache[::reduce_cache]

    
    for i in range(temp_cache.shape[0]):
        height = temp_cache.shape[1]
        width = temp_cache.shape[2]
        
        img = temp_cache[i]
        
        img = cv2.resize(img, (int(width * scale_width), int(height * scale_height)))
        name = f'case{case}_day{day}_slice_{str(i).zfill(4)}.png'

        cv2.imwrite(os.path.join(output_path, name), img)

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
    

def _get_normalization_values(axial_slices_path : str) -> (float, float, float, float):
    imageFiles = []
    for imgFile in os.listdir(axial_slices_path):
        if imgFile.endswith('.png'):
            imageFiles.append(imgFile)

    img = cv2.imread(os.path.join(axial_slices_path, imageFiles[0]), cv2.IMREAD_UNCHANGED)
    sliceCount = len(imageFiles)

    image_cache = []
    for j in range(sliceCount):
        full_path = os.path.join(axial_slices_path, imageFiles[j])
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        image_cache.append(img)

    # To normalize, we want the min and max across every image in the case/day
    # if we don't do it this way, we end up with different colors and strips on our
    # sagittal and coronal images.
    image_cache_array = np.array(image_cache)
    image_cache_array_max = image_cache_array.flatten().max()
    image_cache_array_min = image_cache_array.min()
    image_cache_array_mean = image_cache_array.mean()
    image_cache_array_std = image_cache_array.std()

    return image_cache_array_max, image_cache_array_min, image_cache_array_mean, image_cache_array_std


# Move the images and masks from their subdirectories into 
# one directory as required by the PyTorch ImageFolder class.
def StageDirectories(inDir : str, outDir : str):

    def find_png_files(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".png"):
                    yield os.path.join(root, file)

    os.makedirs(outDir, exist_ok=True)

    file_paths =  [
        (os.path.join(inDir, 'Axial', 'images'), os.path.join(outDir, 'Axial', 'images')),
        (os.path.join(inDir, 'Axial', 'masks'), os.path.join(outDir, 'Axial', 'masks')),
        (os.path.join(inDir, 'Coronal', 'images'), os.path.join(outDir, 'Coronal', 'images')),
        (os.path.join(inDir, 'Coronal', 'masks'), os.path.join(outDir, 'Coronal', 'masks')),
        (os.path.join(inDir, 'Sagittal', 'images'), os.path.join(outDir, 'Sagittal', 'images')),
        (os.path.join(inDir, 'Sagittal', 'masks'), os.path.join(outDir, 'Sagittal', 'masks'))
    ]

    # Get every PNG file in each directory and subdirectory
    # For each file, copy it to the output directory
    for file_path in file_paths:
        inDir = file_path[0]
        outDir = file_path[1]

        os.makedirs(outDir, exist_ok=True)
    
        for file in find_png_files(inDir):
            shutil.copy(file, outDir)

if __name__ == "__main__":
    main()
   
   
    

   #args = parse_args()
   #StageDirectories(args.out_path , args.out_path + '_staged')
    

