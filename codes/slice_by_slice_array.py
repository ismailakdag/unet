# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 08:09:19 2022

@author: caglar.gurkan
"""

import os
import nibabel as ni
import numpy as np
from scipy.ndimage import zoom

#Image'ler 3 channel olacak şekilde ayarlama işlemi---->
path_mask = r"D:\U-Net_Denemeler\deneme_5\data_new/masks/"
path_images = r"D:\U-Net_Denemeler\deneme_5\data_new/imgs/"

path_remask = r"D:\U-Net_Denemeler\deneme_5\org_bak\masks/"
path_reimage = r"D:\U-Net_Denemeler\deneme_5\org_bak\imgs/"

for image in os.listdir(path_images):
    source_image = ni.load(path_images + image)
    ni_image = source_image.get_fdata()
    
    # ni_image_2 = np.uint8(ni_image)          #resimler eksi değerlerinden kurtuldu. 
    
    
    # from skimage import exposure
    # from skimage.util import img_as_ubyte
    # ni_image = exposure.rescale_intensity(ni_image, out_range='uint8')
    # ni_image_2 = img_as_ubyte(ni_image)
    
    
    
    
    
    
    
    
    
    
    # new_hwc = 128                         #resimlerin height ve width değerlerini 128 yap ardından 3 channel olacak şekilde ayarla
    # data_h = new_hwc / ni_image.shape[0]
    # data_w = new_hwc / ni_image.shape[1]
    # data_c = new_hwc / ni_image.shape[2]
    
    # ni_image = zoom(ni_image, (data_h, data_w, 1), order=0)     #128,128,channel oldular
    
    # midchannel_number = ni_image.shape[2] // 2
    
    
    
    
    
    
    
    
    
    
    
    w, h, c = ni_image.shape
    
    for i in range(c):
        
        ni_image_v2 = ni_image[:, :, i]    #128,128,3 oldular
        # ni_image = np.transpose(ni_image, (2, 0, 1))
        
        # source_image = ni.Nifti1Image(ni_image, affine=source_image.affine)   #Nifti image'e dönüşüm yapması gerekmektedir.
        np.save(path_reimage + image + str (i), ni_image_v2)


#MASKELERİN CHANNEL SAYILARINI 1 YAPMA İŞLEMİ----->
for masks in os.listdir(path_mask):
    # print(ni.load(path_mask + masks).shape)
    source_mask = ni.load(path_mask + masks)
    mask = source_mask.get_fdata() 
    
    
    mask = np.int8(mask)          #resimler eksi değerlerinden kurtuldu. 
    
    # new_hwc = 128                         #resimlerin height ve width değerlerini 128 yap ardından 3 channel olacak şekilde ayarla
    # data_h = new_hwc / mask.shape[0]
    # data_w = new_hwc / mask.shape[1]
    # data_c = new_hwc / mask.shape[2]
    
    # mask = zoom(mask, (data_h, data_w, 1), order=0)     #128,128,channel oldular
    

    w, h, c = mask.shape
    
    for i in range(c):
        
        mask_v2 = mask[:, :, i]    #128,128,3 oldular
        # ni_image = np.transpose(ni_image, (2, 0, 1))
        
    
    
        # source_mask = ni.Nifti1Image(mask, affine=source_mask.affine)   #Nifti image'e dönüşüm yapması gerekmektedir.
        np.save(path_remask + masks + str (i), mask_v2)