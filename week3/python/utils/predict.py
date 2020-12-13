# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 22:41:09 2020

@author: User
"""

import numpy as np
from utils.sigmoid import sigmoid
def predict(theta, X):
    
    m = X.shape[0]
    p = np.zeros(m)


    p = np.round(sigmoid(X.dot(theta)))
    return p 