import math
import os

import cv2
import numpy as np
from data.mri_images.csv_data_parser import RecordData
from data.images.image_conversion import reshape_image
from data.mri_images.image_segmentation import generate_segmented_image_24bit_48bit
from data.images.image_utilities import get_image_cache_from_directory


def generate_axial_mri(output_path_image: str, 
                       output_path_mask: str, 
                       case: str, 
                       day: str,
                       data_records : [RecordData], 
                       target_shape_width: int, 
                       target_shape_height: int,
                       img_min,
                       img_max, 
                       post_scaled_mean,
                       post_scaled_stdev):
    
    os.makedirs(output_path_image, exist_ok=True)
    os.makedirs(output_path_mask, exist_ok=True)

    case_records = [record for record in data_records.values() if record.case == case and record.day == day]
    sorted_records = sorted(case_records, key=lambda x: (int(x.case), int(x.day), int(x.slice)))
    for i, slice_record in enumerate(sorted_records):
        
        # Normalization going from 0 to 1 and not upscaling. 
        img = cv2.imread(slice_record.scan_file, cv2.IMREAD_UNCHANGED)
        original_shape = img.shape
        
        img = img.astype(np.float32)
        img = np.clip(img, img_min, img_max)
        img = (img - img_min)/(img_max - img_min)
        img = normalize_pixel_size(img, slice_record.pixel_width, slice_record.pixel_height, slice_record.image_width, slice_record.image_height)
        img = reshape_image(img, target_shape_width, target_shape_height)
       
        
        # # Temp to see what the image would look like when visible. 
        # img = (img - post_scaled_mean) / post_scaled_stdev
        # min = img.min()
        # max=  img.max() 
        # img = ((img - min)/(max - min)) * 255
        # cv2.imwrite(f'{output_path_image}/{i}.png', img)
        
        # This may no longer be between 0 and 1, but that's fine
        # as that range was linearly scalled down and then recentered. 
        # by standard dev. 
        img = (img - post_scaled_mean) / post_scaled_stdev
        name = f'case{case}_day{day}_slice_{str(i).zfill(4)}'
        
        full_path_to_file = os.path.join(output_path_image, name)
        np.save(full_path_to_file, img)
      
        # This should remain the same as our output we want to compare against the masks as is. 
        segmentCollection = generate_segmented_image_24bit_48bit(original_shape, slice_record.organ_list)        
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
    
    cache = get_image_cache_from_directory(axial_slices_path)
    __save_sliced_images(cache, output_path, (1, 0, 2, 3), case, day, scale_height=3)

def generate_axial_from_coronal_mri(coronal_slices_path : str, output_path: str, case: str, day: str):

    cache = get_image_cache_from_directory(coronal_slices_path)
    __save_sliced_images(cache, output_path, (1, 0, 2, 3), case, day, reduce_cache=3)

def generate_sagittal_from_axial_mri(axial_slices_path : str, output_path: str, case: str, day: str):

    cache = get_image_cache_from_directory(axial_slices_path)
    __save_sliced_images(cache, output_path, (2, 1, 0, 3), case, day, scale_width=3)

def generate_axial_from_sagittal_mri(sagittal_slices_path : str, output_path: str, case: str, day: str):

    cache = get_image_cache_from_directory(sagittal_slices_path)
    __save_sliced_images(cache, output_path, (2, 1, 0, 3), case, day, reduce_cache=3)


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


#------------------------------------------------
# Archive:


# def generate_axial_mri_original_resolution(
#     axial_slices_path : str, 
#     output_path_image: str, 
#     output_path_mask: str, 
#     case: str, 
#     day: str,
#     data_records : [RecordData], 
#     target_shape_width: int, 
#     target_shape_height: int,
#     img_min,
#     img_max, 
#     post_scaled_mean,
#     post_scaled_stdev):

#     os.makedirs(output_path_image, exist_ok=True)
#     os.makedirs(output_path_mask, exist_ok=True)

#     case_records = [record for record in data_records.values() if record.case == case and record.day == day]
#     sorted_records = sorted(case_records, key=lambda x: (int(x.case), int(x.day), int(x.slice)))
#     for i, slice_record in enumerate(sorted_records):
#         shape, _, original_color_image_48bit = convert_original_image_to_24bit_48bit(slice_record.scan_file, img_min, img_max)
#         original_color_image_48bit = normalize_pixel_size(original_color_image_48bit, slice_record.pixel_width, slice_record.pixel_height, slice_record.image_width, slice_record.image_height)
#         original_color_image_48bit = reshape_image(original_color_image_48bit, target_shape_width, target_shape_height)

#         name = f'case{case}_day{day}_slice_{str(i).zfill(4)}.png'
#         cv2.imwrite(os.path.join(output_path_image, name), original_color_image_48bit)

#         segmentCollection = generate_segmented_image_24bit_48bit(shape, slice_record.organ_list)        
#         img = normalize_pixel_size(segmentCollection.Color24Bit, slice_record.pixel_width, slice_record.pixel_height, slice_record.image_width, slice_record.image_height)
#         img = reshape_image(img, target_shape_width, target_shape_height)
        
#         name = f'case{case}_day{day}_slice_{str(i).zfill(4)}.png'
#         cv2.imwrite(os.path.join(output_path_mask, name), img)

#     # If we have less than 144 slices, we need to pad the images with black images
#     if len(sorted_records) < 144:
#         print(f'Case {case} day {day} does not have 144 slices. Padding with black images.')
#         blank = np.zeros((int(target_shape_height), int(target_shape_width), 3), dtype=np.uint8)
#         for i in range(len(sorted_records), 144):
#             name = f'case{case}_day{day}_slice_{str(i).zfill(4)}.png'
#             cv2.imwrite(os.path.join(output_path_image, name), blank)
#             cv2.imwrite(os.path.join(output_path_mask, name), blank)


# def process_slice(i, slice_record, case, day, img_min, img_max, target_shape_width, target_shape_height, output_path_image, output_path_mask):
#     shape, original_color_image_24bit, _ = convert_original_image_to_24bit_48bit(slice_record.scan_file, img_min, img_max)
#     original_color_image_24bit = normalize_pixel_size(original_color_image_24bit, slice_record.pixel_width, slice_record.pixel_height, slice_record.image_width, slice_record.image_height)
#     original_color_image_24bit = reshape_image(original_color_image_24bit, target_shape_width, target_shape_height)

#     name = f'case{case}_day{day}_slice_{str(i).zfill(4)}.png'
#     cv2.imwrite(os.path.join(output_path_image, name), original_color_image_24bit)

#     segmentCollection = generate_segmented_image_24bit_48bit(shape, slice_record.organ_list)        
#     img = normalize_pixel_size(segmentCollection.Color24Bit, slice_record.pixel_width, slice_record.pixel_height, slice_record.image_width, slice_record.image_height)
#     img = reshape_image(img, target_shape_width, target_shape_height)
    
#     cv2.imwrite(os.path.join(output_path_mask, name), img)
    


# def _get_normalization_values(axial_slices_path : str) -> (float, float, float, float):
#     ''' Currently this works against one image for one case. The reason being
#         if we don't normalize within a case, the different slices may have the wrong
#         set of colors when adding them together. However, if doing normalization over
#         everything, then this is probably not needed
#     '''
    
#     imageFiles = []
#     for imgFile in os.listdir(axial_slices_path):
#         if imgFile.endswith('.png'):
#             imageFiles.append(imgFile)

#     img = cv2.imread(os.path.join(axial_slices_path, imageFiles[0]), cv2.IMREAD_UNCHANGED)
#     sliceCount = len(imageFiles)

#     image_cache = []
#     for j in range(sliceCount):
#         full_path = os.path.join(axial_slices_path, imageFiles[j])
#         img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
#         image_cache.append(img)

#     # To normalize, we want the min and max across every image in the case/day
#     # if we don't do it this way, we end up with different colors and strips on our
#     # sagittal and coronal images.
#     image_cache_array = np.array(image_cache)
#     image_cache_array_max = image_cache_array.flatten().max()
#     image_cache_array_min = image_cache_array.min()
#     image_cache_array_mean = image_cache_array.mean()
#     image_cache_array_std = image_cache_array.std()

#     return image_cache_array_max, image_cache_array_min, image_cache_array_mean, image_cache_array_std












# #------------------------------------------------
# # Testing code from here down. This may no longer be needed 
# # Since the code for pre processing data has been moved into the 
# # data_module prepare() function.

# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         prog="test_main.py",
#         description="Convert csv file to segmented image.")
    
#     parser.add_argument("--csv_path", type=str, help="Path to segmentation csv file", required=True)
#     parser.add_argument("--base_path", type=str, help="Path to the training data.", required=True)
#     parser.add_argument("--out_path", type=str, help="Path to save segmented image.", required=True)

#     args = parser.parse_args()
#     return args

 
# # Move the images and masks from their subdirectories into 
# # one directory as required by the PyTorch ImageFolder class.
# def StageDirectories(inDir : str, outDir : str):

#     def find_png_files(directory):
#         for root, dirs, files in os.walk(directory):
#             for file in files:
#                 if file.endswith(".png"):
#                     yield os.path.join(root, file)

#     os.makedirs(outDir, exist_ok=True)

#     file_paths =  [
#         (os.path.join(inDir, MriPlane.AXIAL, MriPlane.IMAGES), os.path.join(outDir, MriPlane.AXIAL, MriPlane.IMAGES)),
#         (os.path.join(inDir, MriPlane.AXIAL, MriPlane.MASKS), os.path.join(outDir, MriPlane.AXIAL, MriPlane.MASKS)),
#         (os.path.join(inDir, MriPlane.CORONAL, MriPlane.IMAGES), os.path.join(outDir, MriPlane.CORONAL, MriPlane.IMAGES)),
#         (os.path.join(inDir, MriPlane.CORONAL, MriPlane.MASKS), os.path.join(outDir, MriPlane.CORONAL, MriPlane.MASKS)),
#         (os.path.join(inDir, MriPlane.SAGITTAL, MriPlane.IMAGES), os.path.join(outDir, MriPlane.SAGITTAL, MriPlane.IMAGES)),
#         (os.path.join(inDir, MriPlane.SAGITTAL, MriPlane.MASKS), os.path.join(outDir, MriPlane.SAGITTAL, MriPlane.MASKS))
#     ]

#     # Get every PNG file in each directory and subdirectory
#     # For each file, copy it to the output directory
#     for file_path in file_paths:
#         inDir = file_path[0]
#         outDir = file_path[1]

#         os.makedirs(outDir, exist_ok=True)
    
#         for file in find_png_files(inDir):
#             shutil.copy(file, outDir)

# def main():
#     args = parse_args()

#     train_dir = args.base_path
#     train_csv_path = args.csv_path
#     out_path = args.out_path

#     # get_stats_for_dir(train_dir)
#     # return

#     data_records = parse_csv_into_record_data(train_dir, train_csv_path)
#     if (data_records is None or len(data_records) == 0):
#         print(f"Could not find segmentation data files in {train_dir}")
#         return

#     # every image needs to be normalized to 1 pixel = 1 mm and then reshaped to the maximum
#     # so that all the data is the same physical size and image size. 
#     max_width =  max(map(lambda x: x[1].pixel_width * x[1].image_width, data_records.items()))
#     max_height =  max(map(lambda x: x[1].pixel_width * x[1].image_height, data_records.items()))

#     axial_path = os.path.join(out_path, MriPlane.AXIAL)
#     axial_image_path = os.path.join(axial_path, MriPlane.IMAGES)
#     axial_mask_path = os.path.join(axial_path, MriPlane.MASKS)

#     coronal_path = os.path.join(args.out_path, MriPlane.CORONAL)
#     coronal_image_path = os.path.join(coronal_path, MriPlane.IMAGES)
#     coronal_mask_path = os.path.join(coronal_path, MriPlane.MASKS)

#     sagittal_path = os.path.join(args.out_path, MriPlane.SAGITTAL)
#     sagittal_image_path = os.path.join(sagittal_path, MriPlane.IMAGES)
#     sagittal_mask_path = os.path.join(sagittal_path, MriPlane.MASKS)

#     # Walk through data by Case Number, as we will need to generate images using all the 
#     # slices in that directory.
#     case_numbers = sorted(set(record.case for record in data_records.values()))
#     for case_number in case_numbers:
#         days = sorted(set({record.day for record in data_records.values() if record.case == case_number}))
#         for day in days:

#             # case_records = [record for record in data_recordss() if record.case == case_number and record.day == day]
#             # sorted_records = sorted(case_records, key=lambda x: (int(x.case), int(x.day), int(x.slice)))
#             # if len(sorted_records) != 144:
#             #     print(f'Case {case_number} day {day} does not have 144 slices. Skipping.')
#             case_axial_image_path = os.path.join(axial_image_path, f'case{case_number}_day{day}')
#             case_axial_mask_path = os.path.join(axial_mask_path, f'case{case_number}_day{day}')
#             generate_axial_mri_original_resolution(train_dir, case_axial_image_path, case_axial_mask_path, case_number, day,
#                         data_records, max_width, max_height)

#             # # Generate the coronal MRI images for the specific case and day,
#             # # using the axial images we just generated.
#             # case_coronal_image_path = os.path.join(coronal_image_path, f'case{case_number}_day{day}')
#             # case_coronal_mask_path = os.path.join(coronal_mask_path, f'case{case_number}_day{day}')
#             # generate_coronal_from_axial_mri(case_axial_image_path, case_coronal_image_path, case_number, day)
#             # generate_coronal_from_axial_mri(case_axial_mask_path, case_coronal_mask_path, case_number, day)
                
#             # # Generate the sagittal MRI images for the specific case and day
#             # case_sagittal_image_path = os.path.join(sagittal_image_path, f'case{case_number}_day{day}')           
#             # case_sagittal_mask_path = os.path.join(sagittal_mask_path, f'case{case_number}_day{day}')           
#             # generate_sagittal_from_axial_mri(case_axial_image_path, case_sagittal_image_path, case_number, day)
#             # generate_sagittal_from_axial_mri(case_axial_mask_path, case_sagittal_mask_path, case_number, day)

#     StageDirectories(out_path, out_path + '_staged')

# if __name__ == "__main__":
#     main()
   
   