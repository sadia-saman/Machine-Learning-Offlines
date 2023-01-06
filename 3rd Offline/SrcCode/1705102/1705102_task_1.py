import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as anim 
import functions as func
from scipy.stats import multivariate_normal
import time


#read data from txt file
print("Enter text file name: ")
filename = input()
data = np.loadtxt("../../DataSet/"+filename)

""" #plot data
plt.plot(data[:,0], data[:,1], 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.show() """

#data shape 
N,M = np.shape(data)

k = np.random.randint(1,10)

#intialize parameters
phi, weights, mu, sigma = func.initialize(data,k)

num_iters = 100
log_likelihood = np.zeros(num_iters)

fig = plt.figure()
#plt.scatter(data[:,0], data[:,1], c='b', marker='.', label='data points')

X = np.linspace(data[:,0].min(), data[:,0].max(), np.shape(data)[0])
Y = np.linspace(data[:,1].min(), data[:,1].max(), np.shape(data)[0])
X,Y = np.meshgrid(X,Y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
contours = []

""" for j in range(k):
    rv = multivariate_normal(mu[j], sigma[j])
    plt.contour(X, Y, rv.pdf(pos))  """

print("k = ",k)
plt.ion() 
for i in range(num_iters): 
    posterior_prob,log_likelihood[i]  = func.calculate_posterior_probability(data, mu, sigma, phi, k)
    phi , mu, sigma = func.update_parameters(data, posterior_prob) 
    fig.clear()
    fig.canvas.flush_events() 
    plt.scatter(data[:,0], data[:,1], c='b', marker='.', label='data points')
    for j in range(k): 
        rv = multivariate_normal(mu[j], sigma[j])
        plt.contour(X, Y, rv.pdf(pos)) 
    plt.show()
    


""" plt.plot(log_likelihood)
plt.xlabel('iterations')
plt.ylabel('log likelihood')
plt.show() """











