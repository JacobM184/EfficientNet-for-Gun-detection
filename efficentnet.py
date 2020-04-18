from __future__ import print_function, division
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
from PIL import Image


if (__name__ == '__main__'):
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters
    num_epochs = 1
    num_classes = 2
    batch_size = 20
    learning_rate = 0.001

    #Data transformations
    data_transforms = {
      'train':transforms.Compose([
          transforms.RandomResizedCrop(size=256, scale=(0.75, 1.0)),
          transforms.RandomRotation(degrees=25),
         transforms.RandomHorizontalFlip(),
         transforms.ColorJitter(),


           transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),#Remove if not useful
            transforms.CenterCrop(size=224),#Efficentnet requires 244x244 rgb inputs
          transforms.ToTensor(),
          transforms.RandomErasing(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

          ]),

    'val':    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
       ]),
    }


    data_dir = 'data/'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                      for x in ['train', 'val']}
    test_loader  = torch.utils.data.DataLoader(image_datasets['val'], batch_size=4,
                                                 shuffle=True, num_workers=4)


    train_loader =  torch.utils.data.DataLoader(image_datasets['train'], batch_size=4,
                                                 shuffle=True, num_workers=4)


    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    # Convolutional neural network (two convolutional layers)

    """class ConvNet(nn.Module):
        def __init__(self, num_classes=2):
            super(ConvNet, self).__init__()
            self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,  bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU())


            self.fc = nn.Linear(32, num_classes)

        def forward(self, x):
            out = self.layer1(x)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            return out"""
    class Swish(nn.Module):
        def __init__(self):
            super(Swish, self).__init__()
            self.sigmoid = nn.Sigmoid()

        def forward(self, y):
            return y * self.sigmoid(y)

    def conv1x1(inputCh, outputCh):
    	return nn.Sequential(
        nn.Conv2d(inputCh, outputCh, 1,1,0, bias=False),
        nn.BatchNorm2d(outputCh),
        Swish()
        )
    class DropOutLayer(nn.Module):
        def __init__(self, DropPRate):
            super(DropOutLayer, self).__init__()

            self.DropoutLayer = nn.Dropout2d(p=DropPRate, inplace=True)

        def forward(self, x):
            y = self.DropoutLayer(x)
            return y


    class MBConv(nn.Module):
        def __init__(self, inputCh, outputCh, filterSize, stride, expandRatio, SERatio, DropPRate):
            super(MBConv, self).__init__()
            self.plc = ((stride == 1) and (inputCh == outputCh))
            expandedCh = inputCh * expandRatio
            MBconv = []
            self.use_res = (stride == 1 and (inputCh == outputCh))
            if (expandRatio != 1):
                expansionPhase = nn.Sequential(
                    nn.Conv2d(inputCh, expandedCh, kernel_size=1, bias=False), nn.BatchNorm2d(expandedCh),
                    Swish()
                )
                MBconv.append(expansionPhase)

            DepthwisePhase = nn.Sequential(
                nn.Conv2d(expandedCh, expandedCh, filterSize, stride, filterSize // 2, groups=expandedCh, bias=False),
                nn.BatchNorm2d(expandedCh), Swish()
            )
            MBconv.append(DepthwisePhase)
            hidden_dim = inputCh * expandRatio
            reduced_dim = max(1, int(inputCh / SERatio))
            # insert SqueezeAndExcite here later
            if (SERatio != 0.0):
                SqAndEx = SqueezeAndExcitation(    expandedCh, inputCh, SERatio)
                MBconv.append(SqAndEx)


            projectionPhase = nn.Sequential(
                nn.Conv2d(expandedCh, outputCh, kernel_size=1, bias=False), nn.BatchNorm2d(outputCh)
            )
            MBconv.append(projectionPhase)

            self.MBConvLayers = nn.Sequential(*MBconv)


        def forward(self, x):
            if self.use_res:
                return ( x + self.DropOutLayer(self.MBConvLayers(x)) )
            else:
                return self.MBConvLayers(x)
    ################################# Squeeze and Excite ##################################################################
    ########### Squeeze and Excitation block ###################

    class SqueezeAndExcitation(nn.Module):
        def __init__(self, inputCh, squeezeCh, SERatio):
            super(SqueezeAndExcitation, self).__init__()

            squeezeChannels = int(squeezeCh * SERatio)

            # May have to use AdaptiveAvgPool3d instead, but
            # we need to try this out first in case
            self.GAPooling = nn.AdaptiveAvgPool2d(1)
            self.dense = nn.Sequential(nn.Conv2d(inputCh, squeezeChannels, 1), nn.ReLU(),
                                       nn.Conv2d(squeezeChannels, inputCh, 1), nn.Sigmoid())

        def forward(self, x):
            y = self.GAPooling(x)
            y = self.dense(y)
            return x * y


    class ConvNet(nn.Module):
        def __init__(self, num_classes=2):
            super(ConvNet, self).__init__()

            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                Swish())
            self.mbconv1= MBConv(32,16,3,1,1,0.25,1)

            self.conv1x1 = conv1x1(16,128)
            self.pool= nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc = nn.Linear(112*112*128, num_classes)

        def forward(self, x):
            out = self.layer1(x)
            out = self.mbconv1(out)
            out = self.conv1x1(out)

            out = self.pool(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            return out
    model = ConvNet(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Test the model
    print(model)
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')
