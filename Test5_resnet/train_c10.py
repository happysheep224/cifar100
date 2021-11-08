import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import torchvision.models as models

from model import resnet34,resnet50,resnet101


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

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
    batch_size = 64

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size, shuffle=True,num_workers=4
                                               )

    test_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=batch_size, shuffle=False,num_workers=4
                                                )
    len_train = len(train_data)
    len_test = len(test_data)
    print("using {} images for training, {} images for validation.".format(len_train,
                                                                           len_test))
    for X, y in test_loader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break

    model_weight_path = "./resNet34.pth"
    net = resnet34(num_classes=1000)
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
    net.fc = nn.Linear(in_channel, 100)
    net.to(device)#move to device
    print(net)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 50
    best_acc = 0.0
    save_path = './resNet34-final.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
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


if __name__ == '__main__':
    main()
