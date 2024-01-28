# Standard library imports
import fnmatch
import multiprocessing as mp
import os
import shutil
import zipfile

# Related third party imports
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

# Local application/library specific imports
from common import DataType, MriPlane, TrainingStage, config
from .csv_data_parser import parse_csv_into_record_data
from .generate_working_mri_images import generate_axial_mri, generate_coronal_from_axial_mri, generate_sagittal_from_axial_mri

# This is in the data directory and contains flags
# that let us know what stages of setup have already been 
# completed and can be skipped. 
SETUP_LOG_DIR  = 'setup_log'
TEMP_FOLDER = 'temp'

# This takes the data from the zip file and preprocesses it, restages it
# and does other manipulations that setup the system for training as follows:

# 1. Unzips the mri zip file into a temp directory
# 2. Parses a train.csv file that is expected to be in the base directory of the zip.
# 3. Checks for max width and height across all the records read in from the CSV.
#       The MRI's are at different mm resolutions and vary in size. Using 
#       a combo of the existing width times the mm per pixel we can create a 
#       standard size for all input images and their masks. 
# 4. Splits the downloaded records into 3 sets, training, validation and test. 
# 5. Uses the existing images from temp and runs a series of manipulations on them
#       that adjust sizes, 16bit gray to 8bit, generating masks,etc. 
# 6. Puts those new files into the data directory divided by:
#       train, validation, and test
#       then by axial, coronal, sagittal
#       then by images or masks. 
#       then under each the case0_day0
#    
# During training, the training dataloader can read slices in any order without an issue
# during validation and test, we need to handle the data by case/day. Because hausdorff
# requires a 3d construction of the image. So the dataloader has to act a bit differently.

class MriDataModule(pl.LightningDataModule):
    
    def __init__(self, data_dir, zip_file, mri_plane):
        super().__init__()

        self.data_dir = data_dir
        self.setup_log = os.path.join(self.data_dir, SETUP_LOG_DIR)
        self.working_dir = os.path.join(self.data_dir, TEMP_FOLDER)

        self.axial_train_images = os.path.join(self.data_dir, TrainingStage.TRAIN, mri_plane, DataType.IMAGES)
        self.axial_train_masks = os.path.join(self.data_dir, TrainingStage.TRAIN, mri_plane, DataType.MASKS)

        self.axial_validation_images = os.path.join(self.data_dir, TrainingStage.VALIDATION, mri_plane, DataType.IMAGES)
        self.axial_validation_masks = os.path.join(self.data_dir, TrainingStage.VALIDATION, mri_plane, DataType.MASKS)

        self.axial_test_images = os.path.join(self.data_dir, TrainingStage.TEST, mri_plane, DataType.IMAGES)
        self.axial_test_masks = os.path.join(self.data_dir, TrainingStage.TEST, mri_plane, DataType.MASKS)

        self.axial_minitest_images = os.path.join(self.data_dir, TrainingStage.MINITEST, mri_plane, DataType.IMAGES)
        self.axial_minitest_masks = os.path.join(self.data_dir, TrainingStage.MINITEST, mri_plane, DataType.MASKS)

        self.zip_file = zip_file
        
    def prepare_data(self):
        # Called on single GPU
        # return 
    
        # This will check to see if a specific step has already run
        # and run the necessary steps if it has not. 
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.setup_log, exist_ok=True)

        # Extract the data
        if (not self._has_stage_executed('extract_data')):
            print('about to extract data from zip file.')
            self._extract_data()
            print('unzip complete.')
            self._stage_executed_successfully('extract_data')

        if (not self._has_stage_executed('preprocess_axial_data')):
            print('about to preprocess axial data.')

            # Read the information on each image from the CSV file. 
            zip_images = os.path.join(self.working_dir, config.IMAGE_FOLDER_IN_ZIP)
            zip_csv_file = os.path.join(self.working_dir, config.TRAIN_CSV_IN_ZIP)
            data_records = parse_csv_into_record_data(zip_images, zip_csv_file)
            if (data_records is None or len(data_records) == 0):
                raise Exception(f'No data records found in {self.data_dir}')
            
            # every image needs to be normalized to 1 pixel = 1 mm and then reshaped to the maximum
            # so that all the data is the same physical size and image size. 
            max_width =  max(map(lambda x: x[1].pixel_width * x[1].image_width, data_records.items()))
            max_height =  max(map(lambda x: x[1].pixel_width * x[1].image_height, data_records.items()))

            train_set, validation_set, test_set = \
                self._train_validation_test_split(data_records, config.TRAIN_SIZE, config.VAL_SIZE)

            self._prepare_axial_mri_data_mt(train_set, TrainingStage.TRAIN, max_width, max_height)
            self._prepare_axial_mri_data_mt(validation_set, TrainingStage.VALIDATION, max_width, max_height)
            self._prepare_axial_mri_data_mt(test_set, TrainingStage.TEST, max_width, max_height)

            print('axial data processed.')
            self._stage_executed_successfully('preprocess_axial_data')

        if (not self._has_stage_executed('preprocess_coronal_data')):
            print('about to generate coronal data from axial.')
            
            # Generate the coronal MRI images for the specific case and day,
            # using the axial images we just generated.
            self._generate_coronal_from_axial_mri_mt(TrainingStage.TRAIN)
            self._generate_coronal_from_axial_mri_mt(TrainingStage.VALIDATION)
            self._generate_coronal_from_axial_mri_mt(TrainingStage.TEST)
                
            print('coronal data processed.')
            self._stage_executed_successfully('preprocess_coronal_data')

        if (not self._has_stage_executed('preprocess_sagittal_data')):
            print('about to preprocess sagittal data.')

            # Generate the sagittal MRI images for the specific case and day,
            # using the axial images we just generated.
            self._generate_sagittal_from_axial_mri_mt(TrainingStage.TRAIN)
            self._generate_sagittal_from_axial_mri_mt(TrainingStage.VALIDATION)
            self._generate_sagittal_from_axial_mri_mt(TrainingStage.TEST)
                
            print('sagittal data processed.')
            self._stage_executed_successfully('preprocess_sagittal_data')

    # def setup(self, stage=None):
    #     # Called on every GPU
    #     pass

    ''' Train dataloader is used for training using a mixed batch of images from different cases and days.'''
    def train_dataloader(self):
        return self._get_instance_dataloader(
            self.axial_train_images,
            self.axial_train_masks,
            shuffle=True)
        
    def val_dataloader(self):
        return self._get_instance_dataloader(
            self.axial_validation_images, 
            self.axial_validation_masks)
    
    def test_dataloader(self):
        return self._get_instance_dataloader(
            self.axial_train_images,
            self.axial_train_masks,
            shuffle=True)
        # return self._get_grouped_dataloader(self.axial_test_images, self.axial_test_masks)

    # def predict_dataloader(self):
    #     return self._get_grouped_dataloader(self.axial_test_images, self.axial_test_masks)

    # def minitest_dataloader(self):
    #     return self._get_grouped_dataloader(self.axial_minitest_images, self.axial_minitest_masks)

    def _get_instance_dataloader(self, image_dir, mask_dir, shuffle=False):
        transform = transforms.Compose([transforms.ToTensor()])
        
        instance_dataset = GITractImageInstanceDataset(image_dir, 
                                                       mask_dir, 
                                                       transform)
        
        instance_dataloader = DataLoader(instance_dataset, 
                                      batch_size=16, 
                                      num_workers=31, 
                                      pin_memory=True, 
                                      shuffle=shuffle)
        return instance_dataloader
        

    def _get_grouped_dataloader(self, image_dir, mask_dir):
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = GITractGroupedDataset(image_dir, mask_dir, transform)
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False,
            pin_memory=True)
        return dataloader

    # Determines if a setup step has already been run by checking
    # if a specifically named stage file exists.
    def _has_stage_executed(self, stage: str):
        stage_file = os.path.join(self.setup_log, stage)
        return os.path.exists(stage_file)

    def _stage_executed_successfully(self, stage: str):
        stage_filename = os.path.join(self.setup_log, stage)
        with open(stage_filename, 'w') as file:
            pass

    def _extract_data(self):
        # setup a workspace for the data. We will need that so 
        # that we can stage the data into the dir_data folder 
        # without conflicts from whatever the zip file folder
        # structure is. 
        working_dir = os.path.join(self.data_dir, TEMP_FOLDER)
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)

        os.makedirs(working_dir, exist_ok=True)

        if not os.path.exists(self.zip_file):
            raise FileNotFoundError(f'Could not find zip file {self.zip_file}')
        
        # Extract the data
        with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
            zip_ref.extractall(working_dir)

    def _split_data(self):
        working_dir = os.path.join(self.data_dir, TEMP_FOLDER)

        case_directories = [os.path.join(dirpath, dirnames) for dirpath, dirnames, filenames in os.walk(working_dir) if 'case' in dirnames and 'day' in dirnames]
        case_directories = sorted(case_directories)
        print(case_directories)
        pass


    def _train_validation_test_split(self, data_records, train_size, valid_size):
        # divide the data into train, validation, and test sets based on the case number
        # we have to do this by case, because the slices are not independent.
        case_numbers = sorted(set(record.case for record in data_records.values()))
        
        # Calculate test size
        test_size = 1 - train_size - valid_size
        train_indices, remaining_indices = train_test_split(case_numbers, random_state=42, test_size=(valid_size + test_size))
        valid_ratio = valid_size / (valid_size + test_size)  # Calculate ratio of validation set relative to the original dataset size
        valid_indices, test_indices = train_test_split(remaining_indices, random_state=42, test_size=(1 - valid_ratio))

        # break data_records into train, validation, and test sets
        train_set = {key: value for key, value in data_records.items() if value.case in train_indices}
        valid_set = {key: value for key, value in data_records.items() if value.case in valid_indices}
        test_set = {key: value for key, value in data_records.items() if value.case in test_indices}

        return train_set, valid_set, test_set

    def _prepare_axial_mri_data_mt(self, data_records, subdirectory_name, max_width, max_height):

        # Create directories that the image will ultimately be stored in.
        axial_path = os.path.join(self.data_dir, subdirectory_name, MriPlane.AXIAL)
        os.makedirs(axial_path, exist_ok=True)

        axial_image_path = os.path.join(axial_path, DataType.IMAGES)
        os.makedirs(axial_image_path, exist_ok=True)

        axial_mask_path = os.path.join(axial_path, DataType.MASKS)
        os.makedirs(axial_mask_path, exist_ok=True)

        case_numbers = sorted(set(record.case for record in data_records.values()))
        working_dir = os.path.join(self.data_dir, TEMP_FOLDER, config.IMAGE_FOLDER_IN_ZIP)

        tasks = []
        for case_number in case_numbers:
            days = sorted(set({record.day for record in data_records.values() if record.case == case_number}))
            for day in days:
                case_axial_image_path = os.path.join(axial_image_path, f'case{case_number}_day{day}')
                case_axial_mask_path = os.path.join(axial_mask_path, f'case{case_number}_day{day}')

                # Add the task to the list
                tasks.append((working_dir, case_axial_image_path, case_axial_mask_path, case_number, day, data_records, max_width, max_height))

        # Create a pool of workers and apply generate_axial_mri to each task
        with mp.Pool(mp.cpu_count()) as pool:
            pool.starmap(generate_axial_mri, tasks)


    def _prepare_axial_mri_data_st(self, data_records, subdirectory_name, max_width, max_height):

        # Create directories that the image will ultimately be stored in.
        axial_path = os.path.join(self.data_dir, subdirectory_name, MriPlane.AXIAL)
        os.makedirs(axial_path, exist_ok=True)

        axial_image_path = os.path.join(axial_path, DataType.IMAGES)
        os.makedirs(axial_image_path, exist_ok=True)

        axial_mask_path = os.path.join(axial_path, DataType.MASKS)
        os.makedirs(axial_mask_path, exist_ok=True)

        case_numbers = sorted(set(record.case for record in data_records.values()))
        working_dir = os.path.join(self.data_dir, TEMP_FOLDER, config.IMAGE_FOLDER_IN_ZIP)
        for case_number in case_numbers:
            days = sorted(set({record.day for record in data_records.values() if record.case == case_number}))
            for day in days:
                case_axial_image_path = os.path.join(axial_image_path, f'case{case_number}_day{day}')
                case_axial_mask_path = os.path.join(axial_mask_path, f'case{case_number}_day{day}')

                # This step converts the original images into the 24bit format, as well as converts the 
                # rle in the csv files into a three channel mask image.
                generate_axial_mri(working_dir, case_axial_image_path, case_axial_mask_path, case_number, day,
                            data_records, max_width, max_height)

    def _generate_coronal_from_axial_mri_mt(self, subdirectory_name):
        
        axial_path = os.path.join(self.data_dir, subdirectory_name, MriPlane.AXIAL)
        axial_image_path = os.path.join(axial_path, DataType.IMAGES)
        axial_mask_path = os.path.join(axial_path, DataType.MASKS)
        
        coronal_path = os.path.join(self.data_dir, subdirectory_name, MriPlane.CORONAL)
        os.makedirs(coronal_path, exist_ok=True)

        coronal_image_path = os.path.join(coronal_path, DataType.IMAGES)
        os.makedirs(coronal_image_path, exist_ok=True)

        coronal_mask_path = os.path.join(coronal_path, DataType.MASKS)
        os.makedirs(coronal_mask_path, exist_ok=True)

        tasks = [(name, axial_image_path, axial_mask_path, coronal_image_path, coronal_mask_path, MriPlane.CORONAL) 
                 for name in os.listdir(axial_image_path) 
                 if os.path.isdir(os.path.join(axial_image_path, name))]

        # Create a pool of workers and apply process_case to each task
        with mp.Pool(mp.cpu_count())  as pool:
            pool.starmap(_generate_from_axial_mri_process_case, tasks)

    def _generate_sagittal_from_axial_mri_mt(self, subdirectory_name):
        
        axial_path = os.path.join(self.data_dir, subdirectory_name, MriPlane.AXIAL)
        axial_image_path = os.path.join(axial_path, DataType.IMAGES)
        axial_mask_path = os.path.join(axial_path, DataType.MASKS)
        
        sagittal_path = os.path.join(self.data_dir, subdirectory_name, MriPlane.SAGITTAL)
        os.makedirs(sagittal_path, exist_ok=True)

        sagittal_image_path = os.path.join(sagittal_path, DataType.IMAGES)
        os.makedirs(sagittal_image_path, exist_ok=True)

        sagittal_mask_path = os.path.join(sagittal_path, DataType.MASKS)
        os.makedirs(sagittal_mask_path, exist_ok=True)

        tasks = [(name, axial_image_path, axial_mask_path, sagittal_image_path, sagittal_mask_path, MriPlane.SAGITTAL) 
                 for name in os.listdir(axial_image_path) 
                 if os.path.isdir(os.path.join(axial_image_path, name))]

        # Create a pool of workers and apply process_case to each task
        with mp.Pool(mp.cpu_count())  as pool:
            pool.starmap(_generate_from_axial_mri_process_case, tasks)


def _generate_from_axial_mri_process_case(name, 
                                          axial_image_path, 
                                          axial_mask_path, 
                                          output_image_path, 
                                          output_mask_path,
                                          process_type):
    parts = name.split('_')
    case_number = int(parts[0][4:])  # Remove 'case' and convert to int
    day = int(parts[1][3:])  # Remove 'day' and convert to int
    case_axial_image_path = os.path.join(axial_image_path, name)
    case_axial_mask_path = os.path.join(axial_mask_path, name)
    case_output_image_path = os.path.join(output_image_path, f'case{case_number}_day{day}')
    case_output_mask_path = os.path.join(output_mask_path, f'case{case_number}_day{day}')
    
    if process_type == MriPlane.CORONAL:
        generate_coronal_from_axial_mri(case_axial_image_path, case_output_image_path, case_number, day)
        generate_coronal_from_axial_mri(case_axial_mask_path, case_output_mask_path, case_number, day)
    elif process_type == MriPlane.SAGITTAL:
        generate_sagittal_from_axial_mri(case_axial_image_path, case_output_image_path, case_number, day)
        generate_sagittal_from_axial_mri(case_axial_mask_path, case_output_mask_path, case_number, day)
    else:
        raise ValueError(f"Invalid process_type: {process_type}")


''' 
    For training, the images can be fed in as a batch 
    that crosses cases and days. For inference, the
    images should be fed in as a batch that is all
    from the same case and day. 

    The GITractImageInstanceDataset class is for training.
    The GITractGroupedDataset class is for inference.
'''
class GITractImageInstanceDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([os.path.join(dirpath, f)
                       for dirpath, dirnames, files in os.walk(image_dir)
                       for f in fnmatch.filter(files, '*.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
      
        mask_path = img_path.replace(self.image_dir, self.mask_dir)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
      
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, img_path, mask_path
    
class GITractGroupedDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        cases = os.listdir(image_dir)
        
        cases = [case for case in cases if case.find('case') != -1 and case.find('day') != -1]

        self.case_directories = sorted(cases)

    def __len__(self):
        return len(self.case_directories)

    def __getitem__(self, idx):

        case_dir = self.case_directories[idx]
        case_dir_path = os.path.join(self.image_dir, case_dir)
        # The case directory has a list of filenames,
        # the mask directory has the parallel list of filenames
        # so we parse off the filename and then use that to
        # get the associated mask
        image_filenames = sorted(os.listdir(os.path.join(self.image_dir, case_dir)))

        image_cache = []
        mask_cache = []

        for image_filename in image_filenames:
            image_path = os.path.join(self.image_dir, case_dir, image_filename)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
            if self.transform:
                image = self.transform(image)
            image_cache.append(image) # permute to put the channels last

            mask_path = os.path.join(self.mask_dir, case_dir, image_filename)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"No mask file for image: {image_path}")
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise FileNotFoundError(f"Could not read mask: {mask_path}")
            if self.transform:
                mask = self.transform(mask)
            mask_cache.append(mask)  # permute to put the channels last

        case_image_collection = torch.stack(image_cache)
        case_mask_collection = torch.stack(mask_cache)

        # make 144, 3, 465, 540 (B, C, H, W)
        case_image_collection = case_image_collection.squeeze(0)
        
        # make 3, 144, 465, 540 (C, D, H, W)
        # In this case the D, H, W is a 3d image of the case
        # and the 3 of them are the three different organ masks that
        # will be compared separately.
        case_mask_collection = case_mask_collection.squeeze(0)
        case_mask_collection = case_mask_collection.permute(1, 0, 2, 3)

        return case_image_collection, case_mask_collection, case_dir_path


def main(data_dir, zip_file):
    mdl = MriDataModule(data_dir, zip_file)
    mdl.prepare_data()

if __name__ == '__main__':
    data_dir = '/home/kevin/Documents/lightning-test'
    zip_file = '/home/kevin/Documents/uw-madison-gi-tract-image-segmentation/uw-madison-gi-tract-image-segmentation.zip'
    main(data_dir, zip_file)