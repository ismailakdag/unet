# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 18:35:43 2022

@author: caglar.gurkan
"""





import glob
import numpy as np
import cv2


root_dir =  r'D:\Prostate_Setups\Test_2\b'


for filename in glob.iglob(root_dir + '**/*.jpg', recursive=True):
    print(filename)
    im = cv2.imread(filename)
    
    
    imResize = im[64:192, 64:192]
    
    cv2.imwrite(filename , imResize)