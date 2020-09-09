#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:24:22 2020

@author: paradeisios
"""
import numpy as np

def normalEquation(X,y):
    
    inverse = np.linalg.inv(np.dot(X.T,X))
    theta = np.dot(inverse.dot(X.T),y)

    
    return theta