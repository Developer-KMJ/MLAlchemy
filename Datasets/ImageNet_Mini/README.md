# Preparation of ImageNet Mini Dataset (ILSVRC2012)

The original ImageNet dataset takes up over 100gb of compressed space. This makes the set impractical for most casual experimentation or prototyping. In response, this example offers a practical solution by creating a smaller subset derived from the ImageNet Validation dataset. 

The validation dataset consists of 500,000 images contained within a 6GB tar file. Here's how the ImageNet Mini subset is structured:

1) We narrow down the selection to just 200 classes instead of the full 1000 in ImageNet.
2) For each class, we generate 500 training samples.
3) Additionally, we create 50 validation samples for each class.

Download the dataset files.
----
The `imagenet_mini.py` Python script automates the process of downloading, extracting, and organizing the validation set from the ImageNet website at: https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

```
To use it:

1) Navigate to your project's root directory.
2) Run the script using python imagenet_mini.py.
```

The script will download the validation tar file to the execution location and create the following directory structure:

```
imagenet_mini/ 
├── train/
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   ├── n01440764_10029.JPEG
│   │   └── ...
│   ├── n01443537
│   │   ├── n01443537_10007.JPEG
│   │   ├── n01443537_10014.JPEG
│   │   ├── n01443537_10025.JPEG
│   │   └── ...
├── val/
│   ├── n01440764
│   │   ├── ILSVRC2012_val_00000946.JPEG
│   │   ├── ILSVRC2012_val_00001684.JPEG
│   │   └── ...
│   ├── n01443537
│   │   ├── ILSVRC2012_val_00001269.JPEG
│   │   ├── ILSVRC2012_val_00002327.JPEG
│   │   ├── ILSVRC2012_val_00003510.JPEG
│   │   └── ...
```


Additional Ways to get the ILSRVC dataset
---
The datasets can also be downloaded from Kaggle at:

https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data

--or--

From image-net.org using the following wget commands:
```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate 
```

```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate to 
``````

