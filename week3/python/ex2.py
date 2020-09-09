#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 16:09:42 2020

@author: paradeisios
"""

import numpy as np
from scipy import optimize
from utils.plotData import plotData 
from utils.costFunction import costFunction 
from utils.plotDecisionBoundary import plotDecisionBoundary
from utils.sigmoid import sigmoid
from utils.predict import predict
# Import features matrix and classifications
data = np.loadtxt('ex2data1.txt', delimiter=",")
X = data[:,0:2]
y = data[:,2]


print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples')
plotData(X,y)


# Setup the data matrix appropriately, and add ones for the intercept term

row, col = X.shape

X_intercept = np.ones((row,1))
X_padded = np.hstack((X_intercept,X))

initial_theta = np.zeros(X_padded.shape[1])
cost,grad= costFunction(initial_theta,X_padded,y)

#Compute and display initial cost and gradient
print('Cost at test theta: {:.3f}'.format(cost))
print('Expected cost (approx): 0.693')
print('Gradient at test theta: {:.4f}, {:.4f}, {:.4f}'.format(*grad))
print('Expected gradients (approx): -0.1000, -12.0092, -11.2628')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost,grad = costFunction(test_theta, X_padded, y)

print('Cost at test theta: {:.3f}'.format(cost))
print('Expected cost (approx): 0.218');
print('Gradient at test theta: {:.3f}, {:.3f}, {:.3f}'.format(*grad))
print('Expected gradients (approx): 0.043, 2.566, 2.647')


# Compute optimal parameters using the scipy 
options= {'maxiter': 400}

res = optimize.minimize(costFunction,
                        initial_theta,
                        (X_padded, y),
                        jac=True,
                        method='TNC',
                        options=options)

cost = res.fun
theta = res.x

print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
print('Expected cost (approx): 0.203');

print('Theta:{:.3f}, {:.3f}, {:.3f}'.format(*theta))
print('Expected theta (approx):-25.161, 0.206, 0.201')

# Plot decision boundary
plotDecisionBoundary(theta, X_padded,y)

# Predict probability for a student with score 45 on exam 1 and score 85 on exam 2 

grades = np.array([1, 45, 85])
prob = sigmoid(grades.dot(theta))
print('For a student with scores 45 and 85, we predict an admission probability of {:.3f}'.format(prob))
print('Expected value: 0.775 +/- 0.002')


# Compute accuracy on our training set
p = predict(theta, X_padded)
accuracy = np.mean(y==p)*100
print('Train Accuracy: {}%'.format(accuracy))
print('Expected accuracy (approx): 89.0')


