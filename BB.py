from __future__ import print_function, division
import json
import PIL
from PIL import Image, ImageDraw
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision import transforms
import numpy as np


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
from torch.autograd import Variable
import argparse
import shutil

# Preprocess image
tfms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()
    ])


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
def DropOutLayer(x,DropPRate, training):
 if DropPRate> 0 and training:
     keep_prob = 1 - DropPRate

     mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))

     x.div_(keep_prob)
     x.mul_(mask)

 return x


class MBConv(nn.Module):
 def __init__(self, inputCh, outputCh, filterSize, stride, expandRatio, SERatio, DropPRate):
     super(MBConv, self).__init__()
     self.DropPRate=DropPRate
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

         return ( x + DropOutLayer(self.MBConvLayers(x),self.DropPRate, self.training) )
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
     #res=224x224
     self.layer1 = nn.Sequential(
         nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
         nn.BatchNorm2d(32),
         Swish())
     #stride=1, output res=224x224
     self.mbconv1= MBConv(32,  16,  3, 1, 1, SE, 0)
     #stride=2, output res=112x112
     self.mbconv2= MBConv(16,  24,  3, 2, 6, SE, 0)
     self.mbconv2repeat= MBConv(24,  24,  3, 1, 6, SE, 0)
     #stride=2, output res=56x56
     self.mbconv3= MBConv(24,  40,  5, 2, 6, SE, 0)
     self.mbconv3repeat= MBConv(40,  40,  5, 1, 6, SE, 0)
     #stride=2, output res=28x28
     self.mbconv4= MBConv(40,  80,  3, 2, 6, SE, 0.2)
     self.mbconv4repeat= MBConv(80,  80,  3, 1, 6, SE, 0)
     #stride=1, output res=28x28
     self.mbconv5= MBConv(80,  112, 5, 1, 6, SE, 0.2)
     self.mbconv5repeat= MBConv(112,  112, 5, 1, 6, SE, 0)
     #stride=2, output res=14x14
     self.mbconv6= MBConv(112, 192, 5, 2, 6,SE, 0.2)
     self.mbconv6repeat= MBConv(192, 192, 5, 1, 6, SE, 0)
     #stride=2, output res=7x7
     self.mbconv7= MBConv(192, 320, 3, 1, 6,SE, 0)
     #stride=1, output res=7x7
     self.conv1x1 = conv1x1(320,1280)
     #stride=1, output res=7x7
     self.pool=  nn.AdaptiveAvgPool2d(1)
     self.fc = nn.Linear(1280, num_classes)


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


# Classify
model = torch.load('b0_global.pt').to(device)
model = nn.Sequential(
    model,
    nn.Softmax(1)
)
model.eval()
def get_prob(img):
    img = tfms(img).unsqueeze(0)

    img=img.to(device)
    with torch.no_grad():
        outputs = model(img).to(device)


    # Print predictions

    prob=outputs[0,0].item()


    return prob




# This function checks the predictions for all 5 images and returns the img, prediction and location
def checkImg(model, image, threshold):
    # get initial prediction for the entire image
    initialPred = get_prob(image)

    # set dummy values for outputs
    img = 0
    location = 'None'
    pred = 0

    # check if the prediction for entire image is >= the threshold prediction
    if (initialPred >= threshold):
        # run prediction for top left, top right, bottom left, and bottom right
        img1, predOne = topLeft(model, image, threshold)
        img2, predTwo = topRight(model, image, threshold)
        img3, predThree = bottomLeft(model, image, threshold)
        img4, predFour = bottomRight(model, image, threshold)
    else:
        # return dummy values for bad initial prediction
        return img, location, pred

    # get location of highest prediction
    location = checkHighest(predOne, predTwo, predThree, predFour)

    # get highest prediction's image, prediction


    if (location == 'Top Left'):
        img = img1
        pred = predOne

    elif (location == 'Top Right'):
        img = img2
        pred = predTwo

    elif (location == 'Bottom Left'):
        img = img3
        pred = predThree

    elif (location == 'Bottom Right'):
        img = img4
        pred = predFour

    elif (location == 'None'):
        img = 0
        pred = 0

    return img, location, pred
# gets the top left box

def topLeft(model, image, threshold):
    # get dimensions of image and crop
    horiz, vert = image.size

    # get top left and bottom right coordinates to crop
    coords = (0, 0, int(horiz * 0.66), int(vert * 0.66))
    img1 = image.crop(coords)

    # get prediction for cropped img
    predOne = get_prob(img1)


    if (predOne < threshold):
        predOne = 0;

    return img1, predOne




def topRight(model, image, threshold):
    # get dimensions of image and crop
    horiz, vert = image.size

    # get top left and bottom right coordinates to crop
    coords = (int(horiz * 0.33), 0, horiz, int(vert * 0.66))
    img2 = image.crop(coords)

    # get prediction for cropped img
    predTwo = get_prob(img2)

    if (predTwo < threshold):
        predTwo = 0;

    return img2, predTwo
def bottomLeft(model, image, threshold):

    # get dimensions of image and crop
    horiz, vert = image.size

    # get top left and bottom right coordinates to crop
    coords = (0,vert*0.33,horiz*0.66,480)

    img3 = image.crop(coords)#coords)
    horiz, vert = img3.size

    # get prediction for cropped img
    predThree=get_prob(img3)
 # torch.Size([1, 3, 224, 224])

    #predThree = get_prob(img3)

    if (predThree < threshold):
        predThree = 0;

    return img3, predThree


def bottomRight(model, image, threshold):
    # get dimensions of image and crop
    horiz, vert = image.size

    # get top left and bottom right coordinates to crop
    coords = (int(horiz * 0.33), int(vert * 0.33), horiz, vert)
    img4 = image.crop(coords)

    # get prediction for cropped img
    predFour = get_prob(img4)

    if (predFour < threshold):
        predFour = 0;

    return img4, predFour
"""
image = Image.open('panda.JPG')
i,p=topRight(model,image,0.5)
print(i)
print(p)"""
# function to find highest prediction in list of predictions
def checkHighest(predOne, predTwo, predThree, predFour):
    # make a list from inputs
    pred = [predOne, predTwo, predThree, predFour]

    # set default location
    location = 'None'

    # get biggest value from set
    big = max(pred)

    # get location value for biggest prediction
    if(big==0):
        location = 'None'
    elif (big == pred[0]):
        location = 'Top Left'
    elif (big == pred[1]):
        location = 'Top Right'
    elif (big == pred[2]):
        location = 'Bottom Left'
    elif (big == pred[3]):
        location = 'Bottom Right'



    return location


def updateCoords(location, pos):

    # Note: pos = [x1, y1, x2, y2]
    if (location == 'Top Left'):
        A1 = pos[0]
        A2 = pos[1]
        D1 = (pos[0] + int(0.66 * (pos[2] - pos[0])))
        D2 = (pos[1] + int(0.66 * (pos[3] - pos[1])))
        newPos = [A1, A2, D1, D2]
    elif (location == 'Top Right'):
        A1 = (pos[2] - int((pos[2] - pos[0]) * 0.66))
        A2 = pos[1]
        D1 = pos[2]
        D2 = int(0.66 * (pos[3] - pos[1]))
        newPos = [A1, A2, D1, D2]
    elif (location == 'Bottom Left'):
        A1 = pos[0]
        A2 = (pos[1] + int(0.66 * (pos[3] - pos[1])))
        D1 = (pos[0] + int(0.66 * (pos[2] - pos[1])))
        D2 = pos[3]
        newPos = [A1, A2, D1, D2]
    elif (location == 'Bottom Right'):
        A1 = (pos[0] + int(0.66 * (pos[2] - pos[0])))
        A2 = (pos[1] + int(0.66 * (pos[3] - pos[1])))
        D1 = pos[2]
        D2 = pos[3]
        newPos = [A1, A2, D1, D2]
    elif (location == 'None'):
        newPos = pos

    return newPos

# main function
#imagein=Image.open('panda.JPG')

def runbox(model, path, threshold):
    # open image from the file path
    image = Image.open(path)
    # set initial location value
    location = 'Initial'

    # Insert initial image A and D
    # pos = [x1, y1, x2, y2]
    pos = [0, 0, 540, 480]
    counter = 0

    # loop to keep cropping till predictions worsen
    while (location != 'None'):

        # run checkImg function

        img, location, pred = checkImg(model, image, threshold)


        # update image value for each loop
        if (img != 0):
            image = img
        else:
            break
        # update threshold value for each loop
        if (pred != 0):
            threshold = pred
        else:
            break

        if (location != 'None'):
            pos = updateCoords(location, pos)
            counter = counter + 1
    # [A, B, C, D]
    coords = [(pos[0], pos[1]), (pos[2], pos[1]), (pos[0], pos[3]), (pos[2], pos[3])]

    if (counter != 0):
        return coords, location, threshold
    else:
        return [(-1,-1),(0,0),(0,0),(0,0)], 0,0
print("please input filename of picutre [file must be in same directory as this script]")
path=input()
while(os.path.exists(path)==False):
    print("File does not exist. Please make sure filename includes extension [eg. picture.png, photo.jpg]")
    path=input()
cord,l,thres=runbox(model,path,0.60)


# Display the image


im = Image.open(path)

draw = ImageDraw.Draw(im)



# write to stdout

if(cord[0]!=(-1,-1)):

    print("probablity of gun: ",thres)
    a,b=cord[0]

    c,d=cord[3]

    draw.rectangle([(a,b),(c,d)], fill=None, outline=2)
    im.show()
else:
    print("not detected")
