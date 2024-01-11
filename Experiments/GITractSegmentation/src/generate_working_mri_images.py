
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
            generate_coronal_mri(case_axial_image_path, case_coronal_image_path, case_number, day)
            generate_coronal_mri(case_axial_mask_path, case_coronal_mask_path, case_number, day)
                
            # Generate the sagittal MRI images for the specific case and day
            generate_sagittal_mri(case_axial_image_path, case_sagittal_image_path, case_number, day)
            generate_sagittal_mri(case_axial_mask_path, case_sagittal_mask_path, case_number, day)

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


def generate_sagittal_mri(axial_slices_path : str, output_path: str, case: str, day: str):

    imageFiles = []
    for imgFile in sorted(os.listdir(axial_slices_path)):
        if imgFile.endswith('.png'):
            imageFiles.append(imgFile)

    img = cv2.imread(os.path.join(axial_slices_path, imageFiles[0]))
    width = img.shape[0]
    height = img.shape[1]
    sliceCount = len(imageFiles)

    image_cache = []
    for j_slice_to_width_idx in range(sliceCount):
        full_path = os.path.join(axial_slices_path, imageFiles[j_slice_to_width_idx])
        img = cv2.imread(full_path)
        image_cache.append(img)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # This is slices looking in from the patient's left to right  
    for location_in_current_height_to_read_slice_from in range(height):
        new_image = np.zeros((sliceCount, width, 3))
        for j_slice_to_height_idx in range(sliceCount):
            full_path = os.path.join(axial_slices_path, imageFiles[j_slice_to_height_idx])

            # img = cv2.imread(full_path)
            img = image_cache[j_slice_to_height_idx]
            new_image[j_slice_to_height_idx, :, :] = img[:,location_in_current_height_to_read_slice_from, :]

        new_image = cv2.resize(new_image, (sliceCount * 3, width))
        name = f'case{case}_day{day}_slice_{str(location_in_current_height_to_read_slice_from).zfill(4)}.png'
        cv2.imwrite(os.path.join(output_path, name), new_image)

def generate_coronal_mri(axial_slices_path : str, output_path: str, case: str, day: str):

    imageFiles = []
    for imgFile in sorted(os.listdir(axial_slices_path)):
        if imgFile.endswith('.png'):
            imageFiles.append(imgFile)

    img = cv2.imread(os.path.join(axial_slices_path, imageFiles[0]))
    width = img.shape[0]
    height = img.shape[1]
    sliceCount = len(imageFiles)

    image_cache = []
    for j_slice_to_width_idx in range(sliceCount):
        full_path = os.path.join(axial_slices_path, imageFiles[j_slice_to_width_idx])
        img = cv2.imread(full_path)
        image_cache.append(img)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # This is the top looking down view. 
    for location_in_current_width_to_read_slice_from in range(width):
        new_image = np.zeros((height, sliceCount, 3))
        for j_slice_to_width_idx in range(sliceCount):
            full_path = os.path.join(axial_slices_path, imageFiles[j_slice_to_width_idx])
            # img = cv2.imread(full_path)
            img = image_cache[j_slice_to_width_idx]
            new_image[:, j_slice_to_width_idx, :] = img[location_in_current_width_to_read_slice_from, :, :]

        new_image = cv2.resize(new_image, (height, sliceCount * 3))
        name = f'case{case}_day{day}_slice_{str(location_in_current_width_to_read_slice_from).zfill(4)}.png'
        cv2.imwrite(os.path.join(output_path, name), new_image)

def generate_axial_from_coronal(coronal_slices_path : str, output_path: str, case: str, day: str):

    imageFiles = []
    for imgFile in sorted(os.listdir(coronal_slices_path)):
        if imgFile.endswith('.png'):
            imageFiles.append(imgFile)

    img = cv2.imread(os.path.join(coronal_slices_path, imageFiles[0]))
    height = img.shape[0]
    width = img.shape[1]
    sliceCount = len(imageFiles)

    image_cache = []
    for j_slice in range(sliceCount):
        full_path = os.path.join(coronal_slices_path, imageFiles[j_slice])
        img = cv2.imread(full_path)
        img = cv2.resize(img, (width, int(height/3)))
        image_cache.append(img)

    # After resizing it, we need to re-read the settings.
    height = image_cache[0].shape[0]
    width = image_cache[0].shape[1]
    sliceCount = len(image_cache)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get each new image individually, and read through all the rows or columns needed to
    # make the new image.
    for new_image_index in range(height):

        new_image = np.zeros((sliceCount, width, 3))
        for slice in range(sliceCount):
            img = image_cache[slice]
            
            row = img[new_image_index,:, :]
            new_image[slice, :, :] = row

        name = f'case{case}_day{day}_slice_{str(new_image_index).zfill(4)}.png'
        cv2.imwrite(os.path.join(output_path, name), img)

    

    # for location_in_current_height_to_read_slice_from in range(height):
    #     new_image = np.zeros((sliceCount, width, 3))
    #     for j_slice_to_height_idx in range(sliceCount):
    #         img = image_cache[j_slice_to_height_idx]
    #         new_image[j_slice_to_height_idx, :, :] = img[:,location_in_current_height_to_read_slice_from, :]

    #     new_image = cv2.resize(new_image, (sliceCount, width))
    #     name = f'case{case}_day{day}_slice_{str(location_in_current_height_to_read_slice_from).zfill(4)}.png'
    #     cv2.imwrite(os.path.join(output_path, name), new_image)

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
    

