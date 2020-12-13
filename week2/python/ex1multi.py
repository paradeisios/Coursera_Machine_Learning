
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.featureNormalize import featureNormalize
from utils.gradientDescentMulti import gradientDescentMulti
from utils.normalEquation import normalEquation

print('Loading Data ...\n\n\n')
data = np.loadtxt('ex1data2.txt', delimiter=",")
to_view = pd.read_csv('ex1data2.txt',sep=",",header = None)
X = data[:,0:2]
y = data[:,2]
m = len(y)

print('---------- First 10 examples from the dataset ----------')
print(pd.DataFrame(data).head(10))

### Normalization Process

X_normalized, mu, sigma = featureNormalize(X)

#### Add intercept term to X

X_intercept = np.ones((m,1))

X_padded = np.hstack((X_intercept,X_normalized))

#### Gradient Descent 

alpha = 0.1
num_iters = 400

theta = np.zeros(3)

theta,J_history = gradientDescentMulti(X_padded,y,theta,alpha,num_iters)

# Plot the convergence graph

plt.plot(range(len(J_history)),J_history,'-b', linewidth=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

<<<<<<< HEAD
print('Theta computed from gradient descent: {:.2f}, {:.2f}, {:.2f}'.format(theta[0],theta[1],theta[2]))

=======
print('Theta computed from gradient descent: {}, {}, {}'.format(theta[0], theta[1], theta[2]))
>>>>>>> c399ce74ad607aaf411c4c2779c3a832d944b4e9

# Estimate the price of a 1650 sq-ft, 3 br house
X_features = [1, 1650, 3]
X_features [1:3] = (X_features [1:3] - mu) / sigma
price = np.dot(X_features, theta)

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(price))


#Normal Equations
print('Solving with normal equations...')

data = np.loadtxt('ex1data2.txt', delimiter=",")
X = data[:, :2]
y = data[:, 2]
m = y.size
X = np.concatenate([np.ones((m, 1)), X], axis=1)

theta = normalEquation(X,y)

print('Theta computed from the normal equations: {:.2f}, {:.2f}, {:.2f}'.format(theta[0],theta[1],theta[2]))
X_features = [1, 1650, 3]
X_features [1:3] = (X_features [1:3] - mu) / sigma
price = np.dot(X_features, theta)

print('Predicted price of a 1650 sq-ft, 3 br house (using the normal equation): ${:.0f}'.format(price))
