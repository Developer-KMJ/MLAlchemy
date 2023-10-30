import os
import shutil
from urllib.error import HTTPError
import urllib.request
import tarfile

''' Because the ILSRVC2012 dataset is so large, we will only extract a small subset of it.'''
''' For this example, 200 classes with 500 images each. We will use some of the remaining images as a validation set. '''

base_dir = 'imagenet_mini'
train_dir = 'imagenet_mini/train'
val_dir = 'imagenet_mini/val'

''' Displays the progress of the download in the console
    Uses a closure to keep a static out of the global namespace,
    that can minimize the number of status messages displayed. '''
def download(url, tar_file):
    # Initialize the progress tracking variables
    static_last_percent = -1

    # Define the progress hook function
    def download_progress_hook(count, block_size, total_size):
        nonlocal static_last_percent
        downloaded = count * block_size 
        percent = int((downloaded / total_size) * 100)
        if (percent != static_last_percent):
            print(f"Downloaded:  {percent}% - {int(downloaded/1024**2)}mb of {int(total_size/1024**2)}Mb")
            static_last_percent = percent
    
    # Check if the file already exists
    if os.path.exists(tar_file):
        print("The file already exists, skipping download")
        return True
    
    # Download the file and display progress
    try:
        urllib.request.urlretrieve(url, tar_file, reporthook=download_progress_hook)
        return True
    except urllib.error.URLError as e:
        print(f"Failed to download: {e.reason}")
    except HTTPError as e:
        print(f"Failed to download: {e.code} - {e.msg}")
    except Exception as e:
        print(f"Failed to download: {e}")

    return False

def extract_images(tar_file):
     with tarfile.open(tar_file, 'r') as tar:
        # Loop through the contents of the tar file and extract the first 200 files.
        for i, member in enumerate(tar):
            if not member.isfile():
                continue
  
            # Get the tar file for the category.
            name = member.name.replace('.tar', '')
            new_train_path = os.path.join(train_dir, name)
            if os.path.exists(new_train_path):
                os.remove(new_train_path) if os.path.isfile(new_train_path) else shutil.rmtree(new_train_path)
            os.makedirs(new_train_path)
            
            new_val_path = os.path.join(val_dir, name)
            if os.path.exists(new_val_path):
                os.remove(new_val_path) if os.path.isfile(new_val_path) else shutil.rmtree(new_val_path)
            os.makedirs(new_val_path)

            # Extract the tar file and its contents.
            working_path = os.path.join(base_dir, member.name)
            if os.path.exists(working_path):
                os.remove(working_path) if os.path.isfile(working_path) else shutil.rmtree(working_path)
            tar.extract(member, base_dir)
            with tarfile.open(working_path, 'r') as inner_tar:
                for j, child in enumerate(inner_tar):
                    if not child.isfile():
                        continue
                    
                    # Extract the image file to the appropriate directory.
                    child_path = new_train_path if j < 500 else new_val_path
                    inner_tar.extract(child, child_path)
                    
                    if j > 549:
                        break
            
            os.remove(working_path)

            if i >= 199:
                break

''' Downloads the ILSRVC2012 dataset and extracts it a subdirectory of the current directory '''
if __name__ == '__main__':

    url = 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar'
    tar_file = 'ILSVRC2012_img_val.tar'

    success = download(url, tar_file)
    if not success:
        print("Failed to download dataset")

    extract_images(tar_file)
   


