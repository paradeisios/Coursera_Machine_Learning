#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 17:15:37 2021

@author: paradeisios
"""
import numpy as np
from scipy.optimize import minimize
from utils.lrCostFunction import lrCostFunction

def oneVsAll(X,y,num_labels,l):
    
    row,col = X.shape
    X_0 = np.ones((row,1))
    X   = np.concatenate((X_0,X),axis=1)
    
    y = np.where(y==10,0,y)
    
    all_theta = np.zeros((num_labels, col + 1))
    
    for c in range(num_labels):
        
        print(f"Working on num: {c}")
        
        y = y = np.where(y==c,1,0)
        initial_theta = np.zeros(col+ 1)
        
        res = minimize(lrCostFunction, 
                        initial_theta, 
                        (X, y, l), 
                        jac=True, 
                        method='CG',
                        options={'maxiter': 50}) 
        
        all_theta[c]=res.x
   
    return all_theta