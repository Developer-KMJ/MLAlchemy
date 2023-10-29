# Preparation of ImageNet (ILSVRC2012)

-----------------------------------------------------------------------------
Special thanks to Antoine Broyelle, this directory is a mirror of instructions from 
his github site https://gist.github.com/antoinebrl/7d00d5cb6c95ef194c737392ef7e476a

The datasets can also be downloaded from Kaggle at:
https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data

or directly from image-net.org using the following wget commands:
In addition to the download.php the files can be downloaded directoy using wget:
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate 
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate to download the dataset files.
------------------------------------------------------------------------------

The dataset can be found on the [official website](https://image-net.org/download.php) if you are affiliated with
a research organization. It is also available on Academic torrents.

This script extracts all the images and group them so that folders contain images that belong to the same class.

1. Download the `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar`
2. Download the script `wget https://gist.githubusercontent.com/antoinebrl/7d00d5cb6c95ef194c737392ef7e476a/raw/dc53ad5fcb69dcde2b3e0b9d6f8f99d000ead696/prepare.sh`

3. Run it `./prepare.sh`
4. If the files are not in the same folder you can specify their paths `./prepare.sh ~/Dataset/imagenet/ILSVRC2012_img_train.tar ~/Dataset/imagenet/ILSVRC2012_img_val.tar`

The folder should have the following content:
```
train/
├── n01440764
│   ├── n01440764_10026.JPEG
│   ├── n01440764_10027.JPEG
│   ├── n01440764_10029.JPEG
│   └── ...
├── n01443537
│   ├── n01443537_10007.JPEG
│   ├── n01443537_10014.JPEG
│   ├── n01443537_10025.JPEG
│   └── ...
├── ...
└── ...

val/
├── n01440764
│   ├── ILSVRC2012_val_00000946.JPEG
│   ├── ILSVRC2012_val_00001684.JPEG
│   └── ...
├── n01443537
│   ├── ILSVRC2012_val_00001269.JPEG
│   ├── ILSVRC2012_val_00002327.JPEG
│   ├── ILSVRC2012_val_00003510.JPEG
│   └── ...
├── ...
└── ...
```


