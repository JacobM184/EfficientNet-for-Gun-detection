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
from torch.autograd import Variable
import argparse
import shutil


if (__name__ == '__main__'):

  batch=10
  # Device configuration
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)

  test_loader = torch.utils.data.DataLoader(
    #datatransforms
    datasets.ImageFolder('data/guntest', transform=transforms.Compose([


                        torchvision.transforms.Resize((224), interpolation=2),
                          transforms.RandomResizedCrop(size=224, scale=(1, 1.0)),
                        torchvision.transforms.ToTensor()

                    ])),
    batch_size=batch, shuffle=True,  num_workers=4)
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
          if (device=='cpu'):
              mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
          else:
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
          #self.pool=  nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
          self.pool=  nn.AdaptiveAvgPool2d(1)
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

  #change model name inside quotation marks to test another model
  model = torch.load('b0_global.pt').to(device)
  model = nn.Sequential(
      model,
      nn.Softmax(1)
  )
  model.eval()
  test_acc = 0.0
  count=0
  timestamp1 = time.time()
  #run test batch
  for samples, labels in test_loader:
    with torch.no_grad():
        count+=1
        if (device!='cpu'):
            samples, labels = samples.cuda(), labels.cuda()





        output = model(samples)

        # calculate accuracy
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(labels)
        test_acc += torch.mean(correct.float())

  timestamp2 = time.time()

  print( "This took %.2f seconds" % (timestamp2 - timestamp1))
  print( "time per image= %.3f" %( (timestamp2 - timestamp1)/(count*10)))
  print('Accuracy of the network on {} test images: {}%'.format(count*10, round(test_acc.item()*100.0/len(test_loader), 2)))
