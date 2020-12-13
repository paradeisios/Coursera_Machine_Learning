# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 12:28:23 2020

@author: User
"""

import numpy as np
from utils.sigmoid import sigmoid

def costFunctionReg(theta, X, y, lamda = 1):
    
    # Initialize useful parameters
    m = len(y) 
    J = 0 
    grad = np.zeros(theta.shape)
    
    hypothesis = sigmoid(X.dot(theta))
    regularization_parameter = (lamda / (2 * m)) * np.sum(np.power(theta[1:theta.shape[0]],2))
    J = (1 / m) * np.sum(-y.dot(np.log(hypothesis)) - (1 - y).dot(np.log(1 - hypothesis))) + regularization_parameter
    
    grad = (1 / m) * (hypothesis - y).dot(X)
    grad_parameter = theta[1:grad.shape[0]] * lamda / m
    
    grad[1:grad.shape[0]] = grad[1:grad.shape[0]] +  grad_parameter
    
    return (J,grad)