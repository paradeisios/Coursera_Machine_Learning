#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:24:22 2020

@author: paradeisios
"""
from numpy.linalg import pinv 

def normalEquation(X,y):
    
    theta = pinv (X.T @ X) @ X.T @ y

    
    return theta