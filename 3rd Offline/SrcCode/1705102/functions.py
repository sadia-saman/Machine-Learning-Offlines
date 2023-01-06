import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import seaborn as sns
import time

##............functions for E-step................
def  initialize(data,k):
    # initial weights given to each cluster are stored in phi or P(Ci=j)
    phi = np.full(k, fill_value=1/k)

    # initial weights given to each data point wrt to each cluster or P(Xi/Ci=j)
    weights = np.full(shape = np.shape(data), fill_value=1/k)

    # dataset is divided randomly into k parts of unequal sizes
    random_row = np.random.randint(low=0, high=np.shape(data)[0], size=k)
    mu = [data[i, :] for i in random_row] # initial value of mean of k Gaussians

    # initial value of covariance matrix of k Gaussians
    sigma = [ np.cov(data.T) for _ in range(k) ]
    return phi, weights, mu, sigma







#formula : likelihood = f(x|mu,sigma)
def calculate_likelihood(data, mu, sigma,k):
    N = np.shape(data)[0] 
    likelihood = np.zeros((N,k))  
    for j in range(k):
        distribution = multivariate_normal(mean=mu[j], cov=sigma[j],allow_singular=True)
        likelihood[:, j] = distribution.pdf(data)
    return likelihood

#formula : evidence = sum(likelihood * prior_probability)
def calculate_evidence(likelihood,phi):
    
    N = np.shape(likelihood)[0]
    k = np.shape(likelihood)[1]
    evidence = np.zeros(N)
    for i in range(N):
        for j in range(k):
            evidence[i] = evidence[i] + (likelihood[i][j] * phi[j])
    return evidence

def calculate_log_likelihood(evidence):
    N = np.shape(evidence)[0] 
    log_likelihood = 0
    for i in range(N): 
        log_likelihood = log_likelihood + np.log(evidence[i])
    return log_likelihood

###formula : posterior_probability = (likelihood * prior_probability) / evidence
def calculate_posterior_probability(data, mu, sigma, phi, k):
    likelihood = calculate_likelihood(data, mu, sigma,k)
    evidence = calculate_evidence(likelihood,phi) 

    N = np.shape(data)[0]
    posterior_probability = np.zeros((N,k))
    for i in range(N):
        for j in range(k):
            posterior_probability[i,j] = phi[j] * likelihood[i,j] / evidence[i]
    return posterior_probability,calculate_log_likelihood(evidence)


#............functions for M-step.........

def update_parameters(data,posterior_probability):
    #refer to the formula in main_notes.pdf page 140 M-step sections
    #return parameters are psi, mu, sigma
    # here posterior_probability is w in the formula
    phi = np.mean(posterior_probability, axis=0)
    k = np.shape(posterior_probability)[1]
    mu = np.zeros((k,np.shape(data)[1]))
    sigma = np.zeros((k,np.shape(data)[1],np.shape(data)[1]))
    for j in range(k):
        weights = posterior_probability[:,j]

        total_weights = np.sum(weights)
        temp = np.zeros(np.shape(data))
        for i in range(np.shape(data)[0]):
            for m in range(np.shape(data)[1]):
                temp[i][m] = data[i][m] * posterior_probability[i][j]
        mu[j] = np.sum(temp , axis=0)/total_weights
        sigma[j] = np.cov(data.T, aweights=weights, bias=True)

    
    
    return phi, mu, sigma

#............visualize gaussian mixture model......... 

        
    


