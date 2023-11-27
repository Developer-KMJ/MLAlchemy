import torchvision
from torchvision import transforms
from dataset import PetsDataModule
import torch
from torch.utils.data import DataLoader


def batch_mean_and_sd(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (
                      cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
                            cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
      snd_moment - fst_moment ** 2)        
    return mean,std

# pdm = PetsDataModule(data_dir='./data', batch_size=32, num_workers=4)
# pdm.prepare_data()

standard_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor()])

image_data = torchvision.datasets.ImageFolder(root='./data/train', transform=standard_transform)

image_data_loader = DataLoader(
  image_data, 
  batch_size=32)

mean, std = batch_mean_and_sd(image_data_loader)
print("mean and std: \n", mean, std)
