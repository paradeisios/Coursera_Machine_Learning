import numpy as np


def computeCost(theta,X,y):
    
    m = y.shape[0] 
    J = 0
    
    hypothesis = X.dot(theta) - y
    J = (1.0/(2.0*m)) * np.sum(np.square(hypothesis))
    
    return J
    