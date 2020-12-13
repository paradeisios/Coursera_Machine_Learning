# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:48:04 2020

@author: Paradeisios
"""
from utils.computeCost import computeCost
import numpy as np

def gradientDescent(X,y,theta,alpha, iterations):
    
    m = len(y)
    J_history = np.zeros((iterations,1))
    
    derivative = 0
    for i in range(iterations):
        derivative = (X.dot(theta) - y).dot(X)
        theta = theta - alpha*(1/m)*derivative
        
        J_history[i]= computeCost(theta,X,y)
        
    return(theta,J_history)
    
    
