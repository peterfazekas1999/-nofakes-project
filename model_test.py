# -*- coding: utf-8 -*-
import argparse
import os
import sys
import torch
from PIL import Image
from networks.drn_seg import DRNSub
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
import torchvision.transforms as transforms

#this script runs a picture through the network we're training

model_pth = r"C:\Users\paull\Desktop\no fakes\model.pth"   #import DRNSub(1) model

img_path = r"C:\Users\paull\Desktop\no fakes\val\original\flickr_0001.png"


def load_classifier(model_path, gpu_id):                        #loading function from FAL library (global_classifier script)
    if torch.cuda.is_available() and gpu_id != -1:                 
        device = 'cuda:{}'.format(gpu_id)
    else:
        device = 'cpu'
    model = DRNSub(1)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.device = device
    model.eval()
    return model




tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])

#load model and picture

model = load_classifier(model_pth,0)

pic = Image.open(img_path).convert('RGB')

pic = tf(pic).to(model.device)

#run picture through model and print the output

print(model(pic.unsqueeze(0))[0].sigmoid().cpu().item())

