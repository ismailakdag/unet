# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:12:22 2022

@author: caglar.gurkan
"""



from skimage.color import rgb2gray
import numpy as np
from PIL import Image
import glob
import os

root_dir =  r'D:\Monkeypox_Segmentation\dataset\dataset_v3\masks_3'


for filename in glob.iglob(root_dir + '**/*.npy', recursive=True):
    print(filename)
    im = np.load(filename)
    # im = rgb2gray(im)
    # imResize = im.resize((128,128), Image.ANTIALIAS)
    
    # im = np.asarray(im)

    im1 = im/255
    # grayscale = rgb2gray(imResize)
    im[im1>0.5]=1
    im[im1<=0.5]=0
    # result = Image.fromarray((im).astype(np.uint8))
    
    bol = filename.split(".")
    yeni_isim = bol[0] + '.' + bol [1]
    
    np.save(yeni_isim, im)
    
    # os.remove(filename)
    
    
    
    