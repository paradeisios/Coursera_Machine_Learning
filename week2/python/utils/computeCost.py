import numpy as np

def computeCost(theta,X,y):
    m = len(y)
    hypothesis = (X.dot(np.transpose(theta)) - y)
    J = (1.0/(2.0*m)) * np.sum(np.square(hypothesis))
    
    return(J)

