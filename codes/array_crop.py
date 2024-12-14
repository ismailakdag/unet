# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 17:33:56 2022

@author: caglar.gurkan
"""


from PIL import Image
import os, sys


from PIL import Image
import os, sys
import glob

import numpy as np

from tempfile import TemporaryFile

import cv2


root_dir =  r'D:\Prostate_U-Nets\unet\datasets\crop+z-score\images'


for filename in glob.iglob(root_dir + '**/*.npy', recursive=True):
    print(filename)
    im = np.load(filename)
    # box = (150, 150, 350, 350)
    imResize = im[128:384, 128:384]
    
    np.save(filename, imResize)