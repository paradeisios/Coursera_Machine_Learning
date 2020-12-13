# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 18:38:34 2020

@author: User
"""

import numpy as np
from utils.plotData import plotData 
from utils.mapFeature import mapFeature
from matplotlib import pyplot as plt

def plotDecisionBoundary(theta, X, y):
    
    plotData(X[:, 1:3], y)
    
    if X.shape[1]<= 3:
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])
        plot_y = (-1 / theta[2])* (theta[1]*plot_x + theta[0])
        
        plt.plot(plot_x,plot_y)
        
    else:
         u = np.linspace(-1, 1.5, 50)
         v = np.linspace(-1, 1.5, 50)

         z = np.zeros((len(u),len(v)))
         
         for ii,ui in enumerate(u):
             for jj,vj in enumerate(v):
                 z[ii, jj] = np.dot(mapFeature(ui, vj), theta)
                 
         z = z.T
    
         plt.contour(u, v, z, levels=[0], linewidths=2, colors='g')
         plt.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap='Greens', alpha=0.4)
