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
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

# Local application/library specific imports
from common import DataType, MriPlane, TrainingStage, config
from data.images.image_normalization import get_image_normalization_values_records
from .mri_images.csv_data_parser import parse_csv_into_record_data
from data.mri_images.mri_image_generation import generate_axial_mri

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

        self.train_images = os.path.join(self.data_dir, TrainingStage.TRAIN, mri_plane, DataType.IMAGES)
        self.train_masks = os.path.join(self.data_dir, TrainingStage.TRAIN, mri_plane, DataType.MASKS)

        self.validation_images = os.path.join(self.data_dir, TrainingStage.VALIDATION, mri_plane, DataType.IMAGES)
        self.validation_masks = os.path.join(self.data_dir, TrainingStage.VALIDATION, mri_plane, DataType.MASKS)

        self.test_images = os.path.join(self.data_dir, TrainingStage.VALIDATION, mri_plane, DataType.IMAGES)
        self.test_masks = os.path.join(self.data_dir, TrainingStage.VALIDATION, mri_plane, DataType.MASKS)

        self.minitest_images = os.path.join(self.data_dir, TrainingStage.MINITEST, mri_plane, DataType.IMAGES)
        self.minitest_masks = os.path.join(self.data_dir, TrainingStage.MINITEST, mri_plane, DataType.MASKS)

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

            # In this case we are splitting the data we have into three sets, train, validation, and test. So we only want to calculate the 
            # normalization values on the test images, after we have split them so that we don't leak information that could make
            # the validation or test results appear to be better than they are. 
            min_val, max_val, mean, original_stddev, original_variance, mean_scaled, scaled_stddev, scaled_variance = get_image_normalization_values_records(train_set)
            

            self._prepare_axial_mri_data(train_set, TrainingStage.TRAIN, max_width, max_height, min_val, max_val, mean_scaled, scaled_stddev)
            self._prepare_axial_mri_data(validation_set, TrainingStage.VALIDATION, max_width, max_height, min_val, max_val, mean_scaled, scaled_stddev)
            self._prepare_axial_mri_data(test_set, TrainingStage.TEST, max_width, max_height, min_val, max_val, mean_scaled, scaled_stddev)

            print('axial data processed.')
            self._stage_executed_successfully('preprocess_axial_data')

    # def setup(self, stage=None):
    #     # Called on every GPU
    #     pass

    ''' Train dataloader is used for training using a mixed batch of images from different cases and days.'''
    def train_dataloader(self):
        return self._get_instance_dataloader(
            self.train_images,
            self.train_masks,
            shuffle=True)
        
    def val_dataloader(self):
        return self._get_instance_dataloader(
            self.validation_images, 
            self.validation_masks)
    
    def test_dataloader(self):
        #return self._get_instance_dataloader(self.axial_train_images, self.axial_train_masks, shuffle=True)
        return self._get_grouped_dataloader(self.test_images, self.test_masks)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_grouped_dataloader(self.test_images, self.test_masks)

    def _get_instance_dataloader(self, image_dir, mask_dir, shuffle=False):
        # Images were transformed in the original import
        # transform = transforms.Compose([transforms.ToTensor(),
        #                                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        #                                                     std=[0.229, 0.224, 0.225])])
        
        instance_dataset = GITractImageInstanceDataset(image_dir, 
                                                       mask_dir)
        
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

    def _prepare_axial_mri_data(self, 
                                   data_records, 
                                   subdirectory_name, 
                                   max_width, 
                                   max_height,
                                   min_value,
                                   max_value, 
                                   post_scaled_mean,
                                   post_scaled_stdev,
                                   multithreaded=True):

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

                if multithreaded:
                    # Add the task to the list
                    tasks.append((
                        case_axial_image_path, 
                        case_axial_mask_path, 
                        case_number, 
                        day, 
                        data_records, 
                        max_width, 
                        max_height,  
                        min_value,
                        max_value, 
                        post_scaled_mean,
                        post_scaled_stdev))
                    
                   
                else:
                    # This step converts the original images into the 24bit format, as well as converts the 
                    # rle in the csv files into a three channel mask image.
                    generate_axial_mri(
                        case_axial_image_path, 
                        case_axial_mask_path, 
                        case_number, 
                        day,
                        data_records, 
                        max_width, 
                        max_height,
                        min_value,
                        max_value, 
                        post_scaled_mean,
                        post_scaled_stdev)
                    
        if multithreaded:
            # Create a pool of workers and apply generate_axial_mri to each task
            with mp.Pool(mp.cpu_count()) as pool:
                pool.starmap(generate_axial_mri, tasks)
''' 
    For training, the images can be fed in as a batch 
    that crosses cases and days. For inference, the
    images should be fed in as a batch that is alladd
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
                       for f in fnmatch.filter(files, '*.npy')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]
        mask_path = img_path.replace(self.image_dir, self.mask_dir).replace('.npy', '.png')
        
        if (img_path.find('.npy') == -1):
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        else:
            # Using numpy, we load 3 images (2.5 dimensions) 
            # instead of an image with 3 channels.
            lower = idx-1 if idx-1 >= 0 else 0
            upper = idx+1 if idx+1 < len(self.images) else len(self.images)-1
            
            lower_channel = np.load(self.images[lower]).astype(np.float32)
            middle_channel = np.load(self.images[idx]).astype(np.float32)
            upper_channel = np.load(self.images[upper]).astype(np.float32)
            
            image = np.stack([lower_channel, middle_channel, upper_channel], axis=0)
            # image = np.transpose(image, (1, 2, 0)) # Change to (H, W, C)
            
            
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)    
        mask = (mask / 255.0).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1))
      
        # if self.transform:
        #     if (img_path.find('.npy') == -1):
        #         image = self.transform(image)
        #     mask = self.transform(mask)
      
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