# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:15:51 2022

@author: caglar.gurkan
"""

from PIL import Image
import os, sys


from PIL import Image
import os, sys
import glob

import numpy as np

from tempfile import TemporaryFile

root_dir =  r'D:\U-Net_Denemeler\deneme_4\Pytorch-UNet-master\data_v4\masks'


for filename in glob.iglob(root_dir + '**/*.npy', recursive=True):
   
    
    seg = np.load(filename)
    seg = np.resize(seg,(512,512))
    labels =np.unique(seg) # [ 0  1  2  3  4  8 10 11 56]
    num_labels = len(labels) # 9
    segD = np.zeros((seg.shape[0], seg.shape[1], num_labels))
    for i in range(0, num_labels-1):  # this loop starts from label 1 to ignore background 0
        segD[:, :, i] = seg == labels[i]
        
    
    np.save(filename, segD)
