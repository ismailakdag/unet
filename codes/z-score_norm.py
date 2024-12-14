# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:15:13 2022

@author: caglar.gurkan
"""

from PIL import Image
import os, sys


from PIL import Image
import os, sys
import glob

import numpy as np

from tempfile import TemporaryFile

root_dir =  r'D:\Prostate_U-Nets\T2_peripheral_1\unet\datasets\i_2'


for filename in glob.iglob(root_dir + '**/*.npy', recursive=True):
    print(filename)
    image = np.load(filename)
    # box = (150, 150, 350, 350)
    
    
    
    mean = np.mean(image)
    std = np.std(image)
    # Scaling
    image_normalized = (image - mean + 0.000001) / (std  + 0.000001)
            

    np.save(filename, image_normalized)
