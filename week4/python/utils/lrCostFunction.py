#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 14:51:29 2021

@author: paradeisios
"""
import numpy as np
from utils.sigmoid import sigmoid 

def lrCostFunction(theta, X, y, l):
          
    m = len(y)

    
    hypothesis = sigmoid(X @ theta)
    reg_parameter = (l/(2*m)) * np.sum(np.power(theta[1:], 2))
    
    J = (1/m) * np.sum((-(y.T) @ np.log(hypothesis)) - (1-y.T)@(np.log(1-hypothesis))) + reg_parameter
    
    grad = (1/m) * X.T @ (hypothesis-y) 
    grad_parameter = theta[1:] * (l/m)
    grad[1:] += grad_parameter
    
    return J,grad

  