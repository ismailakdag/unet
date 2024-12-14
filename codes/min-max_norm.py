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

root_dir =  r'D:\Monkeypox_Segmentation\dataset\dataset_v3\images_3/'


for filename in glob.iglob(root_dir + '**/*.npy', recursive=True):
    print(filename)
    image = np.load(filename)
    # box = (150, 150, 350, 350)
    
    
    
    
    
    # Identify minimum and maximum
    
    image_normalized = image/image.max()
            

    np.save(filename, image_normalized)


