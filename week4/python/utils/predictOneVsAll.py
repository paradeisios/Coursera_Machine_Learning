#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 20:00:54 2021

@author: paradeisios
"""

import numpy as np
from utils.sigmoid import sigmoid 

def predictOneVsAll(all_theta, X):
    
    row,col = X.shape
    X_0 = np.ones((row,1))
    X   = np.concatenate((X_0,X),axis=1)
    
    hypothesis = sigmoid(X @ all_theta.T)
    p = np.argmax(hypothesis,axis=1)
    
    return p