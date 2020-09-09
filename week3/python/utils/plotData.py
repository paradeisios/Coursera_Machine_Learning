#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 16:17:52 2020

@author: paradeisios
"""
from matplotlib import pyplot as plt
import numpy as np

def plotData(X,y):
    
    # Create figure
    fig, ax = plt.subplots()
    
    # Find Indices of Positive and Negative Examples
    pos = np.where(y == 1)  
    neg = np.where(y == 0) 
    
    # PLot the X features vectors, indexed by positive and negative indexes 
    ax.plot(X[pos[0], 0], X[pos[0], 1], 'bo', marker='+', label='Admitted')
    ax.plot(X[neg[0], 0], X[neg[0], 1], 'ro', marker='o', label='Not Admitted')
    
    # Set x,y axis labels
    ax.set_xlabel('Exam 1 score')
    ax.set_ylabel('Exam 2 score')
    ax.legend()
    
    fig
    
    