import os
import json
import tqdm #进度条
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))

data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
train_data = datasets.CIFAR100(
    root="data",
    train=True,
    download=True,
    transform=data_transform["train"],
)

# Download test data from open datasets.
test_data = datasets.CIFAR100(
    root="data",
    train=False,
    download=True,
    transform=data_transform["val"],
)

batch_size = 128

# Create data loaders.
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

len_train = len(train_data)
len_test = len(test_data)
print("Length of train data: ",len_train)
print("length of test data: ", len_test)
if __name__ == '__main__':
    main()
