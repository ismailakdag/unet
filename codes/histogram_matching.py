# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 15:16:33 2022

@author: caglar.gurkan
"""

import matplotlib.pyplot as plt

from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms

import numpy as np

reference = np.load(r'D:\U-Net_Denemeler\deneme_5(vanilla-unet)\org_bak\On_Islemsiz\Images\1 (78).npy')

    
mean = np.mean(reference)
std = np.std(reference)
# Scaling
reference = (reference - mean + 0.000001) / (std  + 0.000001)


image = np.load(r'D:\U-Net_Denemeler\deneme_5(vanilla-unet)\org_bak\On_Islemsiz\Images\1 (78).npy') / 255

matched = match_histograms(image, reference, channel_axis=-1)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)
for aa in (ax1, ax2, ax3):
    aa.set_axis_off()

ax1.imshow(image)
ax1.set_title('Source')
ax2.imshow(reference)
ax2.set_title('Reference')
ax3.imshow(matched)
ax3.set_title('Matched')

plt.tight_layout()
plt.show()