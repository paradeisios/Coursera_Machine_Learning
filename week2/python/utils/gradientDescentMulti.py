# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:35:29 2020

@author: Paradeisios
"""

import numpy as np
from utils.computeCostMulti import computeCostMulti
from utils.computeCost import computeCost 

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    
    for ii in range(num_iters):
       
        derivative = (X.dot(theta) - y).dot(X)
        theta = theta - alpha*(1/m)*derivative
        
        J_history[ii]= computeCostMulti(theta,X,y)
        derivative = X.T.dot(X.dot(theta) - y)
        theta = theta - alpha*(1.0/m) * derivative
        
        J_history[ii]= computeCost(theta,X,y)
    
    return(theta,J_history)

