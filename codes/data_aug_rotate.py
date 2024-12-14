# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 08:57:10 2022

@author: caglar.gurkan
"""


import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A

import numpy as np

image = np.load(r'D:\U-Net_Denemeler\deneme_5\AUG_DENEME\i/1 (1).npy')
mask = np.load(r'D:\U-Net_Denemeler\deneme_5\AUG_DENEME\m/1 (1)_mask.npy')

original_height, original_width = image.shape[:2]



def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
        
        
        
aug = A.RandomRotate90(p=1)

random.seed(11)
augmented = aug(image=image, mask=mask)

image_heavy = augmented['image']
mask_heavy = augmented['mask']

visualize(image_heavy, mask_heavy, original_image=image, original_mask=mask)

# np.save(r'D:\U-Net_Denemeler\deneme_5\AUG_DENEME\i/1 (2).npy', image_heavy)

# np.save(r'D:\U-Net_Denemeler\deneme_5\AUG_DENEME\m/1 (2)_mask.npy', mask_heavy)
