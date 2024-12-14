# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:12:21 2022

@author: caglar.gurkan
"""






from PIL import Image
import os, sys


from PIL import Image
import os, sys
import glob

root_dir =  r'D:\Prostate_Deneme\PROSTATEx_masks\DENEME_BAK_ONEMLI\val_mask'


for filename in glob.iglob(root_dir + '**/*.jpg', recursive=True):
    print(filename)
    im = Image.open(filename)
    imResize = im.resize((320,480), Image.ANTIALIAS)
    
    imResize.save(filename , 'PNG', quality=90)