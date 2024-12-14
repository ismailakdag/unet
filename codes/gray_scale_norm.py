# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:12:22 2022

@author: caglar.gurkan
"""



from PIL import Image
import os, sys


from PIL import Image
import os, sys
import glob

import numpy as np

from tempfile import TemporaryFile

root_dir =  r'D:\Monkeypox_Segmentation\dataset\dataset_v5\images_2'


for filename in glob.iglob(root_dir + '**/*.npy', recursive=True):
    print(filename)
    image = np.load(filename)
    # box = (150, 150, 350, 350)
    
    
    
    
    
    # Identify minimum and maximum
    max_value = np.max(image)
    min_value = np.min(image)
    # Scaling
    image_scaled = (image - min_value + 0.000001) / \
                            (max_value - min_value + 0.000001)
    # image_normalized = np.around(image_scaled * 255, decimals=0)
            


    np.save(filename, image_scaled)


