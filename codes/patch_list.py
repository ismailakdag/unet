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
from patchify import patchify
from tempfile import TemporaryFile

root_dir =  r'D:\Monkeypox_Segmentation\dataset\dataset_v3\masks_2/'

save_dir =  r'D:\Monkeypox_Segmentation\dataset\dataset_v3\masks_3/'



for filename in os.listdir(root_dir):
    # print(ni.load(path_mask + masks).shape)
    im = np.load(root_dir + filename)

    # box = (150, 150, 350, 350)
    
    patches_img = patchify(im, (112,112), step=56)
    
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, :, :]
            np.save(save_dir + filename + 'image_' + '_'+ str(i)+str(j), single_patch_img)

    