#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 22:01:05 2021

@author: paradeisios
"""
from utils.sigmoid import sigmoid
import numpy as np

def predict(t1,t2,X):
    
    row,_ = X.shape
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    
    a1 = sigmoid(X @ t1.T)
    pad = np.ones((row,1))
    a1 = np.concatenate((pad,a1),axis=1)
    
    a2 = sigmoid(a1 @ t2.T)
    p  = np.argmax(a2,axis=1)
    
    return p