# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:05:54 2020

@author: peter fazekas
"""
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
#import the 2 images
frame1 = cv2.imread('original.jpg')
frame2 = cv2.imread('modified.jpg')
plt.imshow(frame2)
plt.show()
print("frame1 has shape:",frame1.shape)
shape =frame1.shape
factor =3
frame1 = cv2.resize(frame1, (int(shape[1]/factor),int(shape[0]/factor)), interpolation = cv2.INTER_AREA)
frame2 = cv2.resize(frame2, (int(shape[1]/factor),int(shape[0]/factor)), interpolation = cv2.INTER_AREA)
#cv2.imshow('frame2')
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
nexts = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
def get_heatmap_cv(img, magn, max_flow_mag):
    min_flow_mag = .5
    cv_magn = np.clip(
        255 * (magn - min_flow_mag) / (max_flow_mag - min_flow_mag),
        a_min=0,
        a_max=255).astype(np.uint8)
    if img.dtype != np.uint8:
        img = (255 * img).astype(np.uint8)
    
    heatmap_img = cv2.applyColorMap(cv_magn, cv2.COLORMAP_JET)
    heatmap_img = heatmap_img[..., ::-1]
    h, w = magn.shape
    img_alpha = np.ones((h, w), dtype=np.double)[:, :, None]
    heatmap_alpha = np.clip(
        magn / max_flow_mag, a_min=0, a_max=1)[:, :, None]**.7
    heatmap_alpha[heatmap_alpha < .2]**.5
    pm_hm = heatmap_img * heatmap_alpha
    pm_img = img * img_alpha
    cv_out = pm_hm + pm_img * (1 - heatmap_alpha)
    cv_out = np.clip(cv_out, a_min=0, a_max=255).astype(np.uint8)
    
    return cv_out

def save_heatmap_cv(img, magn, path, max_flow_mag=7):
    cv_out = get_heatmap_cv(img, magn, max_flow_mag)
    out = Image.fromarray(cv_out)
    plt.imshow(out)
    plt.show()
    out.save(path, quality=95)
    

nexts = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

#calculates the flow and magnitude needed to create heatmap
flow = cv2.calcOpticalFlowFarneback(prvs, nexts,None, 0.5, 3, 20, 2, 3, 1.2, 0)
print(flow.shape)
mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

path = "tester.png"
tester = save_heatmap_cv(frame1, mag, path)
    