#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 12:39:12 2020

@author: paradeisios
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d(theta0, theta1, J_vals):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(theta0, theta1, J_vals, cmap = 'viridis')
    plt.xlabel('Theta 0')
    plt.ylabel('Theta 1')
    plt.title('Gradient Descent') 


    
def contour_plot(theta, theta0, theta1, J_vals):
     plt.contour(theta0, theta1, J_vals, cmap = 'RdGy',levels = np.logspace(-2, 3, 20))
     plt.xlabel('Theta 0')
     plt.ylabel('Theta 1')
     plt.title('Contour Plot')
     plt.plot(theta[0],theta[1],'rx')
    