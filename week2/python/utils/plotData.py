# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:03:50 2020

@author: User
"""

def plotData(x,y):
    import matplotlib.pyplot as plt
    plt.plot(x,y,'rx',markersize = 10, label = 'Training data')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()
    