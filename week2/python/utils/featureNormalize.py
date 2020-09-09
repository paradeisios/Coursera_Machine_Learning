# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:17:03 2020

@author: Paradeisios
"""

import numpy as np

def featureNormalize(X):
    
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu) / sigma
    
        
    return X_norm, mu, sigma
