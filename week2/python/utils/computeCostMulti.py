#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:23:07 2020

@author: paradeisios
"""

import numpy as np

def computeCostMulti(theta,X,y):
    
    m = y.shape[0] 
    J = 0
    
    hypothesis = X.dot(theta) - y
    J = (1.0/(2.0*m)) * np.sum(np.square(hypothesis))
    
    return J
    
    