import os
import json
import tqdm #进度条
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))


if __name__ == '__main__':
    main()