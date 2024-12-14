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

root_dir =  r'D:\U-Net_Denemeler\deneme_5\org_bak\Masks'


for filename in glob.iglob(root_dir + '**/*.npy', recursive=True):
    print(filename)
    im = np.load(filename)
    # box = (150, 150, 350, 350)
    imResize = im[120:360, 120:360]
    np.save(filename, imResize)