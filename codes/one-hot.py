# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:02:28 2022

@author: caglar.gurkan
"""

import nibabel as nib
import os
from scipy.ndimage import zoom
import numpy as np


seg_file = r"D:\U-Net_Denemeler\deneme_4\Pytorch-UNet-master\data_v4\masks\1 (51)_mask.npy"

seg = np.load(seg_file)
# seg = seg_3d.get_data() # fixme
labels =np.unique(seg) # [ 0  1  2  3  4  8 10 11 56]
num_labels = len(labels) # 9
segD = np.zeros((seg.shape[0], seg.shape[1], num_labels))
for i in range(0, num_labels-1):  # this loop starts from label 1 to ignore background 0
    segD[:, :, i] = seg == labels[i]
    
    