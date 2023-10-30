# Preparation of ImageNet Mini Dataset (ILSVRC2012)

The original ImageNet dataset occupies an extensive 144 gigabytes of compressed space, rendering it impractical for casual experimentation or prototyping. In response to this challenge, our example provides a practical solution: : the curation of a more manageable subset derived from the ImageNet Validation dataset, constituting approximately 10% of the original data.

Recognizing that the validation set alone lacks the requisite volume of data for our experiments, our script takes the necessary step of retrieving the complete training set. Subsequently, it extracts a specific subset represenatitive of the larger dataset.

The ImageNet Mini subset structure
---

1) We narrow down the selection to just 200 classes instead of the full 1000 in ImageNet.
2) For each class, we select 500 training samples.
3) Additionally, we select 50 validation samples for each class.

Download
----
The `imagenet_mini.py` Python script automates the process of downloading, extracting, labeling and organizing the validation set from the ImageNet website.

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
│   │   ├── n01440764_26.JPEG
│   │   ├── n01440764_27.JPEG
│   │   ├── n01440764_29.JPEG
│   │   └── ...
│   ├── n01443537
│   │   ├── n01443537_7.JPEG
│   │   ├── n01443537_14.JPEG
│   │   ├── n01443537_25.JPEG
│   │   └── ...
├── val/
│   ├── n01440764
│   │   ├── n01440764_946.JPEG
│   │   ├── n01440764_1684.JPEG
│   │   └── ...
│   ├── n01443537
│   │   ├── n01443537_1269.JPEG
│   │   ├── n01443537_2327.JPEG
│   │   ├── n01443537_3510.JPEG
│   │   └── ...
```


Additional Ways to get the ILSRVC dataset
---
The datasets can also be downloaded from Kaggle at:

https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data

Or directly from image-net.org using the following wget commands:
```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate 
```

```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate to 
``````

