from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms

if (__name__ == '__main__'):

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # TensorBoard code
    %load_ext tensorboard
    logs_base_dir = "runs"
    os.makedirs(logs_base_dir, exist_ok=True)

    # Hyper parameters
    num_epochs = 0
    num_classes = 2
    batch = 3
    learning_rate = 0.01


    def save_ckp(state):
      f_path = 'checkpoint.pt'
      torch.save(state, f_path)

    ################################################### Data transformations #########################################################

    # Various transformations for training set
    trainingTransforms = transforms.Compose([transforms.RandomRotation(degrees=25),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ColorJitter(),
                                             transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
                                             transforms.Resize((240, 240), interpolation=2),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                             ])
    # Various transformations for validation set
    validTransforms = transforms.Compose([transforms.Resize((240,240), interpolation=2),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          ])

    ################################################### Dataset loading #########################################################

    # set directory for data
    data = "/content/drive/My Drive/data"

    # get training data from the 'train' sub-directory and load the data
    train_set = datasets.ImageFolder(data + "/train", transform = trainingTransforms)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch, shuffle=True, num_workers=4)

    # get testing data from the 'val' sub-directory and load the data
    test_set = datasets.ImageFolder(data + "/val", transform = validTransforms)
  
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch, shuffle=False, num_workers=4)

    # print lengths of data
    print("test set length: ",len(test_set))
    print("test set batch length: ",len(test_loader))
    print("train set length: ",len(train_set))
    print("train set batch length: ",len(train_loader))
    

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
              nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
              nn.BatchNorm2d(32),
              Swish())
          
          #(inputCh, outputCh, filterSize, stride, expandRatio, SERatio, DropPRate)
          #stride=1
          self.mbconv1= MBConv(32,  16,  3, 1, 1, 0.25, 0.2)
          #stride=2, repeat stride = 1
          self.mbconv2= MBConv(16,  24,  3, 2, 6, 0.25, 0.2)
          self.mbconv2repeat= MBConv(24,  24,  3, 1, 6, 0.25, 0.2)
          #stride=2, repeat stride = 1
          self.mbconv3= MBConv(24,  40, 5, 2, 6, 0.25, 0.2)
          self.mbconv3repeat= MBConv(40, 40,  5, 1, 6, 0.25, 0.2)
          #stride=2, repeat stride = 1
          self.mbconv4= MBConv(40,  80, 3, 2, 6, 0.25, 0.2)
          self.mbconv4repeat= MBConv(80, 80,  3, 1, 6, 0.25, 0.2)
          #stride=1, repeat stride = 1
          self.mbconv5= MBConv(80,  112, 5, 1, 6, 0.25, 0.2)
          self.mbconv5repeat= MBConv(112,  112, 5, 1, 6, 0.25, 0.2)
          #stride=2, repeat stride = 1
          self.mbconv6= MBConv(112, 192, 5, 2, 6, 0.25, 0.2)
          self.mbconv6repeat= MBConv(192, 192, 5, 1, 6, 0.25, 0.2)
          #stride=2
          self.mbconv7= MBConv(192, 320, 3, 1, 6, 0.25, 0.2)
          #stride=1
          self.conv1x1 = conv1x1(320,1280)
          #stride=1
          self.pool=  nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
          #8x8x1280 tensor input to two node output
          self.fc = nn.Linear(8*8*1280, num_classes)


      def forward(self, x):
          # layer sequence for EfficientNet B1
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

    ################################################ Tensorboard ########################################################################
    tb = SummaryWriter()
    inputs, labels = next(iter(train_loader))
    network = ConvNet()
    grid = torchvision.utils.make_grid(inputs)
    tb.add_image('images', grid, 0)
    tb.add_graph(network, inputs)
    tb.close()
    ################################################## Training/Eval Code ###############################################################

    # Define model
    model = ConvNet(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    # using nesterov seems to be giving good results i.e. slightly better convergence rate with 0.5 momentum
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5, nesterov=True)

     # LR Scheduler to change rate when convergence plateaus for 15 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15, verbose=True, threshold=0.0001, 
                                                     threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    # code to restart training from checkpoint
    # restart must be set to 1 when required
    restart=1
    if(restart):
          checkpoint = torch.load('/content/drive/My Drive/b1.pt')
          model.load_state_dict(checkpoint['state_dict'])
          optimizer.load_state_dict(checkpoint['optimizer'])
          epoch = checkpoint['epoch']
          print("restarting!")
          print(epoch)
          
    # Train the model
    total_step = len(train_loader)
    model.train()
    for epoch in range(num_epochs):
        # reset variables for accuracy calc
        correct = 0
        total = 0

        #print epoch num
        print('Epoch: ', (epoch + 1))
        model.train()
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
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

            # print out results
            if (i+1) % 500 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Training Accuracy: {:.3f}'
                        .format(epoch+1, num_epochs, i+1, total_step, loss.item(), (100 * correct / total)))
        
        # set checkpoint variables
        checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
        }

        # TensorBoard code to add line graphs for loss, correct guesses, and training accuracy
        tb.add_scalar('Training Loss', loss.item(), epoch)
        tb.add_scalar('Number Correct', correct, epoch)
        tb.add_scalar('Training Accuracy', (100 * correct / total), epoch)

        # Adding histograms for weights, biases and gradients to TensorBoard
        #tb.add_histogram('mbconv1 bias', model.mbconv1.bias, epoch)
        #tb.add_histogram('mbconv1 weight', model.mbconv1.weight, epoch)
        #tb.add_histogram('mbconv1 weight gradients', model.mbconv1.weight.grad, epoch)

        #model_save_name = 'b1.pt'
        path = "/content/drive/My Drive/b1.pt" 
        torch.save(checkpoint, path)

        # testing loop
        # can turn off by changing if(1) to if(0) as required
        if(1):
           # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
           model.eval()  
           with torch.no_grad():
            correct = 0
            total = 0

            # similar structure to trainig loop
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # print out testing results
            print('Test Accuracy: {:.2f} %, Test Loss: {:.4f}'.format((100 * correct / total), loss.item())) 

            # adding testing accuracy to TensorBoard
            tb.add_scalar('Testing Accuracy', (100 * correct / total), epoch)
            tb.add_scalar('Testing Loss', loss.item(), epoch)
            scheduler.step(correct / total)

    # Evaluate the model

    # Confusion matrix code

    # function to plot confusion matrix
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    predlist=torch.zeros(0,dtype=torch.long, device='cuda:0')
    lbllist=torch.zeros(0,dtype=torch.long, device='cuda:0')

    # get predictions from model
    with torch.no_grad():
      for i, (inputs, labels) in enumerate(test_loader, 0):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        predlist = torch.cat([predlist,predicted.view(-1)])
        lbllist=torch.cat([lbllist,labels.view(-1)])


    predlist = predlist.to('cpu')
    lbllist = lbllist.to('cpu')

    # create confusion matrix using sklearn
    c_mtrx = confusion_matrix(lbllist, predlist)

    # labels for data
    names = ('gun', 'not gun')

    # create plot of size 2x2
    plt.figure(figsize=(2,2))

    # plot the confusion matrix
    plot_confusion_matrix(c_mtrx, names)

    # get precision, recall and f1
    print(classification_report(lbllist, predlist))

  

    # Save the final model checkpoint 
    torch.save(model.state_dict(), 'model.ckpt')
