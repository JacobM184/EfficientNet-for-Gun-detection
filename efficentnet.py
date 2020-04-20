from __future__ import print_function, division
import os
import copy # unused
import time # unused
import torch
import torchvision # unused
import numpy as np # unused
import pandas as pd # unused
import torch.nn as nn
from PIL import Image # unused
import torch.optim as optim
import matplotlib.pyplot as plt # unused
from torch.optim import lr_scheduler # unused
import torchvision.transforms as transforms # unused
from torchvision import datasets, models, transforms



if (__name__ == '__main__'):
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters
    num_epochs = 1
    num_classes = 2
    batch_size = 20
    learning_rate = 0.001

################################################### Data transformations #########################################################

    data_transforms = {
        
      # transformations for training set  
      'train': transforms.Compose([
          # random resizing of image up to resolution of 256x256 pixels
          # and scale between 0.75 and 1.0
          transforms.RandomResizedCrop(size=256, scale=(0.75, 1.0)),
          # random rotation of image by +/-25 degrees
          transforms.RandomRotation(degrees=25),
          # random flip transformation on horizontal axis
          transforms.RandomHorizontalFlip(),
          # randomly changes the brightness, contrast and saturation of an image.
          transforms.ColorJitter(),
          # randomly performs perspective transformation of the image randomly with a given probability.
          transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3), # Remove if not useful
          # EfficentNet requires 244x244 RGB inputs, so we crop to the required size
          transforms.CenterCrop(size=224), 
          # converts image to tensor
          transforms.ToTensor(),
          # randomly selects a rectangle region in an image and erases its pixels.
          transforms.RandomErasing(),
          # normalizes a tensor image with mean and standard deviation.
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
          ]),
        
      # transformations for validation set
      'val': transforms.Compose([
          # random resizing of image up to resolution of 256x256 pixels
          transforms.Resize(size=256),
          # EfficentNet requires 244x244 RGB inputs, so we crop to the required size
          transforms.CenterCrop(size=224),
          # converts image to tensor
          transforms.ToTensor(),
          # normalizes a tensor image with mean and standard deviation.
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
          ]),
    }


################################################### Dataset loading #########################################################

    data_dir = 'data/'
    # get training and validation data from dataset in data_dir
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                      for x in ['train', 'val']}
    
    # define testing/validation data
    test_loader  = torch.utils.data.DataLoader(image_datasets['val'], batch_size=4,
                                                 shuffle=True, num_workers=4)

    # define training data
    train_loader =  torch.utils.data.DataLoader(image_datasets['train'], batch_size=4,
                                                 shuffle=True, num_workers=4)

    # store size values for training and testing sets
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    # get the classes in training set
    class_names = image_datasets['train'].classes
    

################################################### Swish activation #########################################################

    class Swish(nn.Module):
        def __init__(self):
            super(Swish, self).__init__()
            self.sigmoid = nn.Sigmoid()

        def forward(self, y):
            # swish is basically equal to the sigmoid of x multiplied by x
            return y * self.sigmoid(y)
    

#################################################### 1x1 Conv layer ###########################################################

    def conv1x1(inputCh, outputCh):
        # convolutional layer with batch normalisation and swish activation
    	return nn.Sequential(
        nn.Conv2d(inputCh, outputCh, 1,1,0, bias=False),
        nn.BatchNorm2d(outputCh),
        Swish()
        )
    

##################################################### Dropout layer ###########################################################

    class DropOutLayer(nn.Module):
        def __init__(self, DropPRate):
            super(DropOutLayer, self).__init__()
            
            # using PyTorch's inbuilt Dropout2d layer with in-place operation
            self.DropoutLayer = nn.Dropout2d(p=DropPRate, inplace=True)

        def forward(self, x):
            y = self.DropoutLayer(x)
            return y

################################################### Squeeze and Excitation ####################################################

    class SqueezeAndExcitation(nn.Module):
        def __init__(self, inputCh, squeezeCh, SERatio):
            super(SqueezeAndExcitation, self).__init__()

            squeezeChannels = int(squeezeCh * SERatio)

            # Using AdaptiveAvgPool2d to perform global average pooling
            # this layer will give an output of form: 1x1xInputChannels
            self.GAPooling = nn.AdaptiveAvgPool2d(1)
            
            # sequential block adds non-linearity, reduces output channel complexity, 
            # and gives each channel a smooth gating function. Output still of the form: 1x1xInputChannels
            self.dense = nn.Sequential(nn.Conv2d(inputCh, squeezeChannels, 1), nn.ReLU(),
                                       nn.Conv2d(squeezeChannels, inputCh, 1), nn.Sigmoid())

        def forward(self, x):
            y = self.GAPooling(x)
            y = self.dense(y)
            
            # multiply the 1x1xInputChannels result of SqueezeAndExcitation with the original input
            # to add more weighting to the feature maps
            return x * y 

###################################################### MBConv block ###########################################################

    class MBConv(nn.Module):
        def __init__(self, inputCh, outputCh, filterSize, stride, expandRatio, SERatio, DropPRate):
            super(MBConv, self).__init__()
            
            # calculate channels after expansion
            expandedCh = inputCh * expandRatio
            
            # array to hold 'sub layers' of MBConv layer
            MBconv = []
            
            # placeholder to check if dropout layers should be used
            self.use_res = (stride == 1 and (inputCh == outputCh))
            
            # code for expansion phase of MBConv layer
            # occurrence depends on the expand ratio
            if (expandRatio != 1):
                # expansion sequential block
                expansionPhase = nn.Sequential(
                    nn.Conv2d(inputCh, expandedCh, kernel_size=1, bias=False), nn.BatchNorm2d(expandedCh),
                    Swish()
                )
                # appending expansion sequential block to the MBConv array
                MBconv.append(expansionPhase)
            
            # depthwise convolution sequential block
            DepthwisePhase = nn.Sequential(
                nn.Conv2d(expandedCh, expandedCh, filterSize, stride, filterSize // 2, groups=expandedCh, bias=False),
                nn.BatchNorm2d(expandedCh), Swish()
            )
            # appending depthwise convolution sequential block to the MBConv array
            MBconv.append(DepthwisePhase)
            
            # Squeeze and excitation sequential block of MBConv
            if (SERatio != 0.0):
                
                # see SqueezeAndExcitation class for implementation
                SqAndEx = SqueezeAndExcitation(expandedCh, inputCh, SERatio)
                
                # appending Squeeze and excitation sequential block to the MBConv array
                MBconv.append(SqAndEx)

            # projection sequential block of MBConv
            # returns channel number from the expanded channels to the intended output channels
            projectionPhase = nn.Sequential(
                nn.Conv2d(expandedCh, outputCh, kernel_size=1, bias=False), nn.BatchNorm2d(outputCh)
            )
            
            # appending projection sequential block to the MBConv array
            MBconv.append(projectionPhase)
            
            # combining all sequential blocks under one 'master' sequential block
            self.MBConvLayers = nn.Sequential(*MBconv)

        # forward function for MBConv
        def forward(self, x):
            
            #logic to check if we need to use a dropout layer
            if self.use_res:
                # see dropout layer class for implementation
                return ( x + self.DropOutLayer(self.MBConvLayers(x)) )
            else:
                return self.MBConvLayers(x)

############################################## Implementation for training & testing ###########################################
   
    # ConvNet class to hold all layers of the efficient net
    class ConvNet(nn.Module):
        def __init__(self, num_classes=2):
            super(ConvNet, self).__init__()
            
            # Conv3x3 layer
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=2),
                nn.BatchNorm2d(32),
                Swish())
            
            # input params for MBConv: (inputCh, outputCh, filterSize, stride, expandRatio, SERatio, DropPRate)
            # stride=1, output res=224x224
            self.mbconv1= MBConv(32,  16,  3, 1, 1, 0.25, 0.2)
            # stride=2, output res=112x112
            self.mbconv2= MBConv(16,  24,  3, 2, 6, 0.25, 0.2)
            self.mbconv2repeat= MBConv(24,  24,  3, 1, 6, 0.25, 0.2)
            # stride=2, output res=56x56
            self.mbconv3= MBConv(24,  40,  5, 2, 6, 0.25, 0.2)
            self.mbconv3repeat= MBConv(40,  40,  5, 1, 6, 0.25, 0.2)
            # stride=2, output res=28x28
            self.mbconv4= MBConv(40,  80,  3, 2, 6, 0.25, 0.2)
            self.mbconv4repeat= MBConv(80,  80,  3, 1, 6, 0.25, 0.2)
            # stride=1, output res=28x28
            self.mbconv5= MBConv(80,  112, 5, 1, 6, 0.25, 0.2)
            self.mbconv5repeat= MBConv(112,  112, 5, 1, 6, 0.25, 0.2)
            # stride=2, output res=14x14
            self.mbconv6= MBConv(112, 192, 5, 2, 6, 0.25, 0.2)
            self.mbconv6repeat= MBConv(192, 192, 5, 1, 6, 0.25, 0.2)
            # stride=2, output res=7x7
            self.mbconv7= MBConv(192, 320, 3, 1, 6, 0.25, 0.2)
            # stride=1, output res=7x7
            self.conv1x1 = conv1x1(320,1280)
            # stride=1, output res=7x7
            self.pool=  nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.fc = nn.Linear(7*7*1280, num_classes)


        def forward(self, x):
            out = self.layer1(x)

            out = self.mbconv1(out)

            out = self.mbconv2(out)

            out = self.mbconv2repeat(out)

            out = self.mbconv3(out)
            out = self.mbconv3repeat(out)

            out = self.mbconv4(out)
            out = self.mbconv4repeat(out)
            out = self.mbconv4repeat(out)

            out = self.mbconv5(out)
            out = self.mbconv5repeat(out)
            out = self.mbconv5repeat(out)

            out = self.mbconv6(out)
            out = self.mbconv6repeat(out)
            out = self.mbconv6repeat(out)
            out = self.mbconv6repeat(out)

            out = self.mbconv7(out)

            out = self.conv1x1(out)

            #out = torch.mean(out, (2, 3))
            out=self.pool(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            return out
    model = ConvNet(num_classes).to(device)

 

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # dynamic optimizer for better training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
