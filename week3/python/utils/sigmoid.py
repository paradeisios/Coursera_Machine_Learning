#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 16:37:56 2020

@author: paradeisios
"""

import numpy as np

def sigmoid(z):
    
    z = np.array(z)
    g = np.zeros(z.shape)
   
    g = 1 / (1 + np.exp(-z))
    
    return(g)