import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

from model import resnet34,resnet50,resnet101
from scipy.io import loadmat, savemat

class GOU_QI_DATA(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_index = self.images[index]
        img_path = os.path.join(self.root_dir, image_index)
        img = loadmat(img_path)["B"]
        label = img_path.split("/")[-1].split("_")[0]   # 我将每个.mat文件名的GS当做类别label（你可以自行修改你原本的label)
        sample = {"image": img, "label": label}

        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using {} device.".format(device))

        # data_transform = {
        #     "train": transforms.Compose([transforms.RandomResizedCrop(224),
        #                                  transforms.RandomHorizontalFlip(),
        #                                  transforms.ToTensor(),
        #                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        #     "val": transforms.Compose([transforms.Resize(256),
        #                                transforms.CenterCrop(224),
        #                                transforms.ToTensor(),
        #                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
        #
        # train_data = datasets.CIFAR100(
        #     root="data",
        #     train=True,
        #     download=True,
        #     transform=data_transform["train"],
        # )
        #
        # # Download test data from open datasets.
        # test_data = datasets.CIFAR100(
        #     root="data",
        #     train=False,
        #     download=True,
        #     transform=data_transform["val"],
        # )
        # batch_size = 64
        #
        # train_loader = torch.utils.data.DataLoader(train_data,
        #                                            batch_size=batch_size, shuffle=True,num_workers=4
        #                                            )
        #
        # test_loader = torch.utils.data.DataLoader(test_data,
        #                                               batch_size=batch_size, shuffle=False,num_workers=4
        #                                             )
        # len_train = len(train_data)
        # len_test = len(test_data)
        # print("using {} images for training, {} images for validation.".format(len_train,
        #                                                                        len_test))

        data = GOU_QI_DATA("/bak4t/back8t/v1/yangdataset/large_dataset/GS/new_data/", transform=None)

        dataloader = DataLoader(data, batch_size=2, num_workers=4, shuffle=True)
        # for X, y in dataloader:
        #     print("Shape of X [N, C, H, W]: ", X.shape)
        #     print("Shape of y: ", y.shape, y.dtype)
        #     break
        for i_batch, batch_data in enumerate(dataloader):
            print(i_batch)  # 打印batch编号
            print(batch_data['image'].size())  # 打印该batch里面图片的大小
            print(batch_data['label'])  # 打印该batch里面图片的标签
        print("The data is loaded ! ")
        model_weight_path = "./resNet34.pth"
        net = resnet34(inputchannel=3)
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))
        # net.cuda()
        # # load pretrain weights
        # # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
        # model_weight_path = "./resnet34-pre.pth"
        # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        # net.load_state_dict(torch.load(model_weight_path, map_location=device))
        # # for param in net.parameters():
        # #     param.requires_grad = False
        #
        # # change fc layer structure
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, 5)
        net.conv1.in_channels = 728
        net.to(device)  # move to device
        device = "gpu"
        print(net)
        print("The structure of net is ok ! ")

        # define loss function
        loss_function = nn.CrossEntropyLoss()

        # construct an optimizer
        params = [p for p in net.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=0.0001)

        epochs = 50
        best_acc = 0.0
        save_path = './resNet34-final.pth'
        train_steps = len(dataloader)
        for epoch in range(epochs):
            # train
            net.train()
            running_loss = 0.0
            train_bar = tqdm(dataloader)
            for step, data in enumerate(train_bar):
                images, labels = data
                optimizer.zero_grad()
                logits = net(images.to(device))
                # logits = net(images)
                loss = loss_function(logits, labels.to(device))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss)

            # validate
            net.eval()
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                val_bar = tqdm(test_loader)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = net(val_images.to(device))
                    # loss = loss_function(outputs, test_labels)
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                               epochs)

            val_accurate = acc / len_test
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate))

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)

        print('The best accuracy is %s', best_acc)
        print('Finished Training')
