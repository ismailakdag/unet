# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 08:19:55 2022

@author: caglar.gurkan
"""

import numpy as np
  
  
cm = np.random.randint(25, size=(2, 2))




# FP = cm[0][1]
# FN = cm[1][0]
# TP = cm[1][1]
# TN = cm[0][0]


FP = cm.sum(axis=0) - np.diag(cm)  
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
        

print(FP[1], FN[1], TP[1], TN[1])


