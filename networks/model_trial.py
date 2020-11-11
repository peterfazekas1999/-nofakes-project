# -*- coding: utf-8 -*-
import argparse
import os
import sys
import torch
from PIL import Image
from drn_seg import DRNSub
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss,BCELoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import pandas as pd
import numpy as np
import torch.nn as nn
# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt
# for creating validation set
from sklearn.model_selection import train_test_split
# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import data, color

df = pd.read_csv(r"C:\Users\peter fazekas\Desktop\nofakes project\FALdetector-master\train.csv")
df= df.rename(columns = {"flickr_0000.png":"ID","0":"labels"})

#load training images
train_size = 10
trainIMG = []

height =50
width=50
for ind in df.index:
    label = df["labels"][ind]
    imgName = df["ID"][ind]
    if(label ==0):
        path = r"C:/Users/peter fazekas/Desktop/nofakes project/FALdetector-master/val/original"+"/" + str(imgName)
    else:
        path = r"C:/Users/peter fazekas/Desktop/nofakes project/FALdetector-master/val/modified"+"/" + str(imgName)
    #change this to import more images
    if(ind>=train_size):
        break
    img = imread(path,as_gray =False)
    img = resize(img,(height,width))
    img/=255.0
    img = img.astype("float32")
    trainIMG.append(img)
X = np.array(trainIMG)
y = df["labels"][:train_size].values
print(X.shape)
print(y.shape)

train_x = X
train_y = y

# converting training images into torch format
train_x = train_x.reshape(train_size,3, height, width)
train_x  = torch.from_numpy(train_x)

# converting the target into torch format
train_y = train_y.astype(np.long);
train_y = torch.from_numpy(train_y)

model = DRNSub(2)
optimizer = Adam(model.parameters(),lr=0.07)   

criterion =CrossEntropyLoss()

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
 
#print(model)

def train(epoch):
    model.train()
    tr_loss = 0
    # getting the training set
    x_train, y_train = Variable(train_x), Variable(train_y)
    # converting the data into GPU format
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()  
    # clearing the Gradients of the model parameters
    optimizer.zero_grad()
    # prediction for training and validation set
    output_train = model(x_train)
    _,predicted = torch.max(output_train.data,1)
    total = train_size
    correct = (predicted==y_train.long()).sum()
    acc = 100*correct/total
    # computing the training and validation loss
    loss_train = criterion(output_train, y_train.long())
    train_losses.append(loss_train)
    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch%1 == 0:
        # printing the validation loss
        print('Epoch : ',epoch+1, '\t', 'loss :',loss_train,"accuracy: ",acc)

n_epochs = 20
train_losses =[]
correct =0
for epoch in range(n_epochs):
    train(epoch)





