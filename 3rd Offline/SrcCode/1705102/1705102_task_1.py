import numpy as np
import matplotlib.pyplot as plt  
from scipy.stats import multivariate_normal 
from sklearn.decomposition import PCA


import numpy as np
from scipy.stats import multivariate_normal 

##............functions for E-step................
def  initialize(data,k):
    
    phi = np.full(k, fill_value=1/k) # P(Ci=j)  
    weights = np.full(shape = (np.shape(data)[0],k), fill_value=1/k) # P(Xi/Ci=j)
    row = np.random.randint(low=0, high=np.shape(data)[0], size=k) # initial value of mean of k Gaussians
    mu = [data[i, :] for i in row] 
    covariance = np.cov(data[0].T) # initial value of covariance matrix of k Gaussians
    sigma = [ covariance for j in range(k) ]
    return phi, weights, mu, sigma


#formula : likelihood = f(x|mu,sigma)
def calculate_likelihood(data, mu, sigma,k):
    N = np.shape(data)[0] 
    likelihood = np.zeros((N,k))  
    for j in range(k):
        distribution = multivariate_normal(mean=mu[j], cov=sigma[j],allow_singular=True)
        likelihood[:, j] = distribution.pdf(data)
    return likelihood


###formula : posterior_probability = (likelihood * prior_probability) / evidence
def calculate_posterior_probability(data, mu, sigma, phi, k):
    likelihood = calculate_likelihood(data, mu, sigma,k)
    evidence = np.dot(likelihood, phi)  
    posterior_probability = np.zeros((np.shape(data)[0],k))
    for i in range(N):
        for j in range(k):
            posterior_probability[i,j] = phi[j] * likelihood[i,j] / evidence[i]
    return posterior_probability, np.sum(np.log(evidence))


#............functions for M-step.........

def update_parameters(data,posterior_probability): 
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


#read data from txt file
print("Enter text file name: ")
filename = input()
data = np.loadtxt("../../DataSet/"+filename)

#data shape 
N,M = np.shape(data)


num_iters = 100
log_likelihood = np.zeros(10)


for k in range(10): 
    phi, weights, mu, sigma = initialize(data,k+1)
    print("k = ",k+1)
    for i in range(num_iters): 
        posterior_prob,likelihood_k  = calculate_posterior_probability(data, mu, sigma, phi, k+1)
        phi , mu, sigma = update_parameters(data, posterior_prob) 
        log_likelihood[k] = log_likelihood[k] + likelihood_k
    

log_likelihood = log_likelihood/np.shape(data)[0]
plt.plot(log_likelihood)
plt.xlabel('number of clusters')
plt.ylabel('log likelihood')
plt.show()

#.........................plotting GMM for k*............................


fig = plt.figure() 


plt.ion()
if (np.shape(data)[1])==2 :
    X = np.linspace(data[:,0].min(), data[:,0].max(), np.shape(data)[0])
    Y = np.linspace(data[:,1].min(), data[:,1].max(), np.shape(data)[0])
    X,Y = np.meshgrid(X,Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
else:
    pca = PCA(n_components=2)
    data_reduced = pca.fit_transform(data)

    X = np.linspace(data_reduced[:,0].min(), data_reduced[:,0].max(), np.shape(data)[0])
    Y = np.linspace(data_reduced[:,1].min(), data_reduced[:,1].max(), np.shape(data)[0])
    X,Y = np.meshgrid(X,Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y


for j in range(8):
    if log_likelihood[j] >= log_likelihood[j+1] and log_likelihood[j+1]<= log_likelihood[j+2]:
        k = (j+1) +1
        break
    k = 4

phi, weights, mu, sigma = initialize(data,k)
for i in range(num_iters): 
    fig.clear()
    fig.canvas.flush_events()
    plt.title("GMM for k* = "+ str(k))

    if (np.shape(data)[1])==2 : 
        plt.scatter(data[:,0], data[:,1], c='b', marker='.', label='data points')
        for j in range(k): 
            rv = multivariate_normal(mu[j], sigma[j], allow_singular=True)
            plt.contour(X, Y, rv.pdf(pos))  
    else :  
        plt.scatter(data_reduced[:, 0],data_reduced[:, 1], c='b', marker='.', label='data points')
        eigen_vectors = pca.components_ 
        
        for j in range(k): 
            mu_reduced = np.dot(mu[j]-pca.mean_, eigen_vectors.T)
            rv = multivariate_normal(mu_reduced, np.dot(np.dot(eigen_vectors ,sigma[j]), eigen_vectors.T),allow_singular=True)
            plt.contour(X, Y, rv.pdf(pos))  
    fig.show()
    posterior_prob,likelihood_k  = calculate_posterior_probability(data, mu, sigma, phi, k)
    phi , mu, sigma = update_parameters(data, posterior_prob)

print("end of iteration")
#........................................................................
plt.ioff()












