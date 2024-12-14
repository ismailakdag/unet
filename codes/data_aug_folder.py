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


import glob

import numpy as np



# def visualize(image, mask, original_image=None, original_mask=None):
#     fontsize = 18
    
#     if original_image is None and original_mask is None:
#         f, ax = plt.subplots(2, 1, figsize=(8, 8))

#         ax[0].imshow(image)
#         ax[1].imshow(mask)
#     else:
#         f, ax = plt.subplots(2, 2, figsize=(8, 8))

#         ax[0, 0].imshow(original_image)
#         ax[0, 0].set_title('Original image', fontsize=fontsize)
        
#         ax[1, 0].imshow(original_mask)
#         ax[1, 0].set_title('Original mask', fontsize=fontsize)
        
#         ax[0, 1].imshow(image)
#         ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        
#         ax[1, 1].imshow(mask)
#         ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
        
        
        

root_dir_masks =  r'D:\U-Net_Denemeler\deneme_5\org_bak\Cropping+Norm+Aug2\Masks'


filename_masks_list = []


for filename_masks in glob.iglob(root_dir_masks + '**/*.npy', recursive=True):
    # print(filename_masks)
    filename_masks_list.append(filename_masks)






root_dir_images =  r'D:\U-Net_Denemeler\deneme_5\org_bak\Cropping+Norm+Aug2\Images'


filename_images_list = []


for filename_images in glob.iglob(root_dir_images + '**/*.npy', recursive=True):
    # print(filename_images)
    filename_images_list.append(filename_images)



for i, mask_aug, img_aug in zip(range(0,247) ,filename_masks_list,filename_images_list):

    mask_aug = np.load(mask_aug)  
    img_aug = np.load(img_aug)  
    
    
            
    aug = A.Rotate(limit=-30)
    
    
    augmented = aug(image=img_aug, mask=mask_aug)
    
    image_h_flipped = augmented['image']
    mask_h_flipped = augmented['mask']



# visualize(image_h_flipped, mask_h_flipped, original_image=img_aug, original_mask=mask_aug)
   
    np.save(filename_images[:-9] + str(i), image_h_flipped)
    np.save(filename_masks[:-14] + str(i), mask_h_flipped)
    
    
