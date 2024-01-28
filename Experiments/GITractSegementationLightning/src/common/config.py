# Raw Data File locations
DATA_DIR = '/home/kevin/Documents/MRIProject-Working/data/processed'
RAW_DATA = '/home/kevin/Documents/MRIProject-Working/data/raw'
ZIP_FILE = '/home/kevin/Documents/MRIProject-Working/data/raw/uw-madison-gi-tract-image-segmentation.zip'

# Info needed for data processing.

# The zip file containing all the images currently has one CSV at the same level as a 
# directory called train. This may not be the same names used for later testing. 
# so they can be changed here. 
IMAGE_FOLDER_IN_ZIP = 'train'
TRAIN_CSV_IN_ZIP = 'train.csv'

# Since we only have existing working data, we need to split them and chose
# this as the percentages. 
TRAIN_SIZE = 0.7
VAL_SIZE = 0.2
TEST_SIZE = 0.1

# location for training results. 
OUT_DIR = '/home/kevin/Documents/MRIProject-Working/out'

# Training Defaults
TRAIN_BATCH_SIZE = 16
LEARNING_RATE=0.0001

