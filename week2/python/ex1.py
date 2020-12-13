import numpy as np
from matplotlib import pyplot as plt 
from utils.plotData import plotData 
from utils.computeCost import computeCost 
from utils.gradientDescent import gradientDescent 
from utils.plotting_help import plot_3d, contour_plot
#### Cost and Gradient Descent


#### Import Data
print('Loading Data ...\n\n\n')
data = np.loadtxt('ex1data1.txt', delimiter=",")
X = data[:,0]
y = data[:,1]

m = len(y)

#### Plot Data 
plotData (X,y)
X_zero = np.ones((m,1)) #initialize X0
X_padded = np.hstack((X_zero,np.transpose([X]))) #pad features

theta = np.zeros(2)# initialize fitting parameters

iterations = 1500 
alpha = 0.01

#### Test cost function with theta = [0,0]
print('Testing the cost fucntion ...\n')
J = computeCost (theta,X_padded,y)

print('With theta = [0 ; 0]\nCost computed = {:.3f}'.format(J))
print('Expected cost value (approx): 32.07\n\n')

#### Test cost function with theta = [-1,2]
new_theta = np.array((-1,2))
J2 = computeCost (new_theta, X_padded,y)

print('With theta = [-1 ; 2]\nCost computed = {:.3f}'.format(J2))
print('Expected cost value (approx): 54.24\n\n')

# run gradient descent
print('Running Gradient Descent ...')

theta_fit,J_history = gradientDescent(X_padded, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent: {}'.format(theta_fit));
print('Expected theta values (approx): -3.6303, 1.1664\n\n');

#### Test linear fit

plt.plot(X, np.dot(X_padded,theta_fit), label = 'Linear Fit')
plt.show()


#### Predict Studies

predict1 = np.array([1, 3.5]).dot(theta_fit)
print("For population = 35,000, we predict a profit of {:f}".format( float(predict1*10000) ))
predict2 = np.array([1, 7]).dot(theta_fit)
print('For population = 70,000, we predict a profit of {:f}'.format( float(predict2*10000) ))

##### Visualing J(theta_0, theta_1)

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for ii in range(len(theta0_vals)):
    for jj in range(len(theta1_vals)):
        t= np.array([theta0_vals[ii], theta1_vals[jj]])
        J_vals[ii,jj] = computeCost(t, X_padded, y)
        
J_vals = J_vals.T

plot_3d(theta0_vals, theta1_vals, J_vals)
contour_plot(theta_fit, theta0_vals, theta1_vals, J_vals)

