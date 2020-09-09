#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 16:34:51 2020

@author: paradeisios
"""

import numpy as np
from utils.sigmoid import sigmoid
def costFunction(theta,X,y):
    
    # Initialize useful parameters
    m = len(y) 
    J = 0 
    grad = np.zeros(theta.shape)
    
    hypothesis = sigmoid(X.dot(theta))
    J = (1 / m) * np.sum(-y.dot(np.log(hypothesis)) - (1 - y).dot(np.log(1 - hypothesis)))
    grad = (1 / m) * (hypothesis - y).dot(X)
    
    return (J,grad)