from matplotlib import pyplot as plt
import numpy as np
from math import sqrt,floor,ceil


def displayData(X,example_width=None):
    
    m,n = X.shape
    
    if example_width == None:
        example_width = round(sqrt(n))
    
    example_height = int(n / example_width)
    
    disp_rows= int(floor(sqrt(m)))
    disp_cols= int(ceil(m/disp_rows))
    
    fig, ax_array = plt.subplots(disp_rows, disp_cols, figsize=(10,10))
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()
    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_height, order='F'),
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')
    

    
    
    
    
    
    
        
            
            