import os
import random
from urllib.error import HTTPError
import urllib.request
import tarfile
from typing import List

''' Because the ILSRVC2012 dataset is so large, we will only extract a small subset of it.'''
''' For this example, 200 classes with 500 images each. We will use some of the remaining images as a validation set. '''

base_dir = 'imagenet_mini'
train_dir = 'imagenet_mini/train'
val_dir = 'imagenet_mini/val'
test_dir = 'imagenet_mini/test'

''' Displays the progress of the download in the console
    Uses a closure to keep a static out of the global namespace,
    that can minimize the number of status messages displayed. '''

# This function downloads the file at the given URL and saves it to the given
# file path. It returns True if the download was successful, or False otherwise.
def download(url: str, tar_file: str) -> bool:
    # Initialize the progress tracking variables
    static_last_percent: int = -1

    # Define the progress hook function
    def download_progress_hook(count: int, block_size: int, total_size: int) -> None:
        nonlocal static_last_percent

        if total_size == 0:
            if static_last_percent < -1:
                print("Invalid total size, cannot report on download")
            static_last_percent = -2
            return

        downloaded: int = count * block_size
        percent: int = int((downloaded / total_size) * 100)
        if (percent != static_last_percent):
            print(
                f"Downloaded:  {percent}% - {downloaded//1024**2}mb of {total_size//1024**2}Mb")
            static_last_percent = percent

    # Check if the file already exists
    if os.path.exists(tar_file):
        file_size: int = os.path.getsize(tar_file)
        with urllib.request.urlopen(url) as response:
            total_size: int = int(response.headers.get('content-length', 0))
        if file_size == total_size:
            print("The file already exists, skipping download")
            return True
        else:
            print("The file exists but is incomplete, downloading again")

    # Download the file and display progress
    try:
        urllib.request.urlretrieve(
            url, tar_file, reporthook=download_progress_hook)
        return True
    except urllib.error.URLError as e:
        print(f"Failed to download: {e.reason}")
    except HTTPError as e:
        print(f"Failed to download: {e.code} - {e.msg}")
    except Exception as e:
        print(f"Failed to download: {e}")

    return False



def download(url: str, tar_file: str) -> bool:
    # Initialize the progress tracking variables
    static_last_percent: int = -1

    # Define the progress hook function
    def download_progress_hook(count: int, block_size: int, total_size: int) -> None:
        nonlocal static_last_percent

        if total_size == 0:
            if static_last_percent < -1:
                print("Invalid total size, cannot report on download")
            static_last_percent = -2
            return

        downloaded: int = count * block_size
        percent: int = int((downloaded / total_size) * 100)
        if (percent != static_last_percent):
            print(
                f"Downloaded:  {percent}% - {downloaded//1024**2}mb of {total_size//1024**2}Mb")
            static_last_percent = percent

    # Check if the file already exists
    if os.path.exists(tar_file):
        file_size: int = os.path.getsize(tar_file)
        with urllib.request.urlopen(url) as response:
            total_size: int = int(response.headers.get('content-length', 0))
        if file_size == total_size:
            print("The file already exists, skipping download")
            return True
        else:
            print("The file exists but is incomplete, downloading again")

    # Download the file and display progress
    try:
        urllib.request.urlretrieve(
            url, tar_file, reporthook=download_progress_hook)
        return True
    except urllib.error.URLError as e:
        print(f"Failed to download: {e.reason}")
    except HTTPError as e:
        print(f"Failed to download: {e.code} - {e.msg}")
    except Exception as e:
        print(f"Failed to download: {e}")

    return False


def extract_images(tar_file: str) -> None:
    random.seed(42)

    with tarfile.open(tar_file, 'r') as tar:
        # Select 200 random classes from the tar file
        members: List[tarfile.TarInfo] = [
            m for m in tar.getmembers() if m.isfile()]
        members.sort(key=lambda member: member.name)
        member_sample: List[tarfile.TarInfo] = random.sample(members, 200)

        # Loop through the selected members and extract the image files.
        for member in member_sample:

            # Get the category name from the member name.
            name: str = os.path.splitext(os.path.basename(member.name))[0]
            new_train_path: str = os.path.join(train_dir, name)
            new_val_path: str = os.path.join(val_dir, name)
            new_test_path: str = os.path.join(test_dir, name)

            # Extract the inner tar file and its contents.
            with tar.extractfile(member) as f, tarfile.open(fileobj=f, mode='r') as inner_tar:
                inner_members: List[tarfile.TarInfo] = [
                    m for m in inner_tar.getmembers() if m.isfile()]
                inner_members.sort(key=lambda inner_member: inner_member.name)
                inner_member_sample: List[tarfile.TarInfo] = random.sample(
                    inner_members, 575)

                # Extract the image file to the appropriate directory.
                for j, child in enumerate(inner_member_sample):
                    if j < 500:
                        child_path = new_train_path
                    elif j < 550:
                        child_path = new_val_path
                    else:
                        child_path = new_test_path
                    inner_tar.extract(child, child_path)
            print(f"Extracted {name}")
    print("Extraction complete!")


''' Downloads the ILSRVC2012 dataset and extracts it a subdirectory of the current directory '''
if __name__ == '__main__':

    url = 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar'
    tar_file = 'ILSVRC2012_img_train.tar'

    success = download(url, tar_file)
    if download(url, tar_file):
        print("Download complete!")
    else:
        print("Download failed!")

    extract_images(tar_file)
