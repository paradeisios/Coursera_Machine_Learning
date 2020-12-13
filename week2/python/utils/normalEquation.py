#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:24:22 2020

@author: paradeisios
"""
def normalEquation(X,y):
    
    theta = (X.T @ y) / (X.T @ X)

    
    return theta