from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms

if (__name__ == '__main__'):
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Hyper parameters
    num_epochs = 40
    num_classes = 10
    batch = 10
    learning_rate = 0.001

    ################################################### Data transformations #########################################################

    trainingTransforms = transforms.Compose([#transforms.RandomResizedCrop(size=256, scale=(0.75, 1.0)), 
                                             #transforms.RandomRotation(degrees=25),
                                             #transforms.RandomHorizontalFlip(),
                                             #transforms.ColorJitter(),
                                             #transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
                                             #transforms.CenterCrop(size=224),
                                             transforms.Resize(300, interpolation=2),
                                             transforms.ToTensor(),
                                             #transforms.RandomErasing(),
                                             #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                             ])

    validTransforms = transforms.Compose([transforms.Resize(300, interpolation=2),
                                          #transforms.CenterCrop(size=224),
                                          transforms.ToTensor(),
                                          #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          ])

    ################################################### Dataset loading #########################################################

    train_loader = torch.utils.data.DataLoader(dataset=(datasets.CIFAR10(root='./data', train=True, download=True, transform=trainingTransforms)),
                                              batch_size=batch, shuffle=True, num_workers=4)

    
    test_loader = torch.utils.data.DataLoader(dataset=(datasets.CIFAR10(root='./data', train=False, download=True, transform=validTransforms)), 
                                             batch_size=batch, shuffle=False, num_workers=4)

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
            nn.Conv2d(inputCh, outputCh, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputCh),
            Swish()
        )


    ##################################################### Dropout layer ###########################################################
    
    def DropOutLayer(x,DropPRate, training):
      if DropPRate> 0 and training:
          keep_prob = 1 - DropPRate

          mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))

          x.div_(keep_prob)
          x.mul_(mask)

      return x
    

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

            self.DropRate = DropPRate
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
            self.DropOut = nn.Dropout2d(p=self.DropRate, inplace=True)
        # forward function for MBConv
        def forward(self, x):

            # logic to check if we need to use a dropout layer
            if self.use_res:
                # see dropout layer class for implementation
                return (x + DropOutLayer(self.MBConvLayers(x), self.DropRate, self.training))
            else:
                return self.MBConvLayers(x)


    ############################################## Implementation for training & testing ###########################################

    # ConvNet class to hold all layers of the efficient net
    class ConvNet(nn.Module):
      def __init__(self, num_classes=10):
          super(ConvNet, self).__init__()
          #res=224x224
          self.layer1 = nn.Sequential(
              nn.Conv2d(3, 38, kernel_size=3, stride=2, padding=1),
              nn.BatchNorm2d(38),
              Swish())
          
          #(inputCh, outputCh, filterSize, stride, expandRatio, SERatio, DropPRate)
          #stride=1, output res=149
          self.mbconv1= MBConv(38,  19,  3, 1, 1, 0.25, 0.2)
          #stride=2, output res=150 74 
          self.mbconv2= MBConv(19,  28,  3, 2, 6, 0.25, 0.2)
          self.mbconv2repeat= MBConv(28,  28,  3, 1, 6, 0.25, 0.2)
          #stride=2, output res=74 36
          self.mbconv3= MBConv(28,  48, 5, 2, 6, 0.25, 0.2)
          self.mbconv3repeat= MBConv(48, 48,  5, 1, 6, 0.25, 0.2)
          #stride=2, output res=37 17.5
          self.mbconv4= MBConv(48,  96, 3, 2, 6, 0.25, 0.2)
          self.mbconv4repeat= MBConv(96, 96,  3, 1, 6, 0.25, 0.2)
          #stride=1, output res=35 16
          self.mbconv5= MBConv(96,  134, 5, 1, 6, 0.25, 0.2)
          self.mbconv5repeat= MBConv(134,  134, 5, 1, 6, 0.25, 0.2)
          #stride=2, output res=17 7
          self.mbconv6= MBConv(134, 230, 5, 2, 6, 0.25, 0.2)
          self.mbconv6repeat= MBConv(230, 230, 5, 1, 6, 0.25, 0.2)
          #stride=2, output res=16 6
          self.mbconv7= MBConv(230, 384, 3, 1, 6, 0.25, 0.2)
          #stride=1, output res=16 6
          self.conv1x1 = conv1x1(384,1280)
          #stride=1, output res=16 6
          self.pool=  nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
          self.fc = nn.Linear(10*10*1280, num_classes)


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
          out = self.mbconv4repeat(out)

          out = self.mbconv5(out)
          out = self.mbconv5repeat(out)
          out = self.mbconv5repeat(out)
          out = self.mbconv5repeat(out)

          out = self.mbconv6(out)
          out = self.mbconv6repeat(out)
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
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    model.train()
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print(inputs[0].size())
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.3f}'
                        .format(epoch+1, num_epochs, i+1, total_step, loss.item(), (100 * correct / total)))

    # Test the model
    print(model)
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

  # Save the model checkpoint 
  #torch.save(model.state_dict(), 'model.ckpt')
