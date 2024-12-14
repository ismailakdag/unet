# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 23:17:53 2022

@author: caglar.gurkan
"""

import glob
import numpy as np


path1 = r'D:\Prostate_U-Nets\T2_peripheral_1\unet\datasets\patch+128_overlap+z-score\masks'
data1 = []
for filename1 in glob.iglob(path1 + '**/*.npy', recursive=True):

    # print (filename)
    
    im = np.load(filename1)
    
    if im.max() !=0:
    
        data1.append(filename1[83:-9])     
        
    
    
path2 = r'D:\Prostate_U-Nets\T2_peripheral_1\unet\datasets\patch+128_overlap+z-score\images'
data2 = []
for filename2 in glob.iglob(path2 + '**/*.npy', recursive=True):

    data2.append(filename2[84:-4]) 


set_difference = set(data2) - set(data1)
list_difference = list(set_difference)



import os

path3 = r'D:\Prostate_U-Nets\T2_peripheral_1\unet\datasets\patch+128_overlap+z-score\masks'
data3 = []
for filename3 in glob.iglob(path3 + '**/*.npy', recursive=True):

    data3.append(filename3) 
        
    for i in list_difference:
        
        if filename3[83:-9] == i:
            
            # path4 = r'D:\3D_Unet\A/'
            
            file_path = filename3
            os.remove(file_path)
            
            
            
path4 = r'D:\Prostate_U-Nets\T2_peripheral_1\unet\datasets\patch+128_overlap+z-score\images'
data4 = []
for filename4 in glob.iglob(path4 + '**/*.npy', recursive=True):

    data4.append(filename4) 
        
    for i in list_difference:
        
        if filename4[84:-4] == i:
            
            # path4 = r'D:\3D_Unet\A/'
            
            file_path = filename4
            os.remove(file_path)

