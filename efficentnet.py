import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


################ Data transformation section ###################################################################################
image_augmentations = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.75, 1.0)),
        transforms.RandomRotation(degrees=25),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),#Remove if not useful
        transforms.CenterCrop(size=224),#Efficentnet requires 244x244 rgb inputs
        transforms.ToTensor(),
        #       transforms.RandomErasing(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),

    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
       ])}
###########################################################################################################################
        
############################ Dataset loading section ######################################################################
dataDirectory = "hymenoptera_data"  # will change dir later
gunDataset = {i: datasets.ImageFolder(os.path.join(dataDirectory, i), image_augmentations[i]) for i in ['train', 'valid']}

dataLoaders = {i: DataLoader(gunDataset[i], batch_size=4, shuffle=True) for i in ['train', 'valid']}

classNames = gunDataset['train'].classes
datasetSize = {i: len(gunDataset[i]) for i in ['train', 'valid']}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
###########################################################################################################################
