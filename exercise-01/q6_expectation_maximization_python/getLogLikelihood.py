import numpy as np

def multiDimensionalNormal(mean, covariance, value):
    D = len(value)

    normalizationFactor = 1/((2*np.pi)**(D/2)*np.linalg.det(covariance)**(0.5))

    distribution = np.exp(-0.5 * (np.transpose(value-mean) @ np.linalg.inv(covariance) @ (value-mean)))

    return normalizationFactor*distribution

def getLogEntry(means, weights, covariances, value):
    K = means.shape[0]
    
    innerSum = 0
    for k in range(K):
        innerSum += weights[k]*multiDimensionalNormal(means[k], covariances[:,:,k], value)

    return np.log(innerSum)

def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    means = np.array(means)
    weights = np.array(weights)

    sumOverDataPoints = 0
    for n in range(X.shape[0]):
        sumOverDataPoints += getLogEntry(means, weights, covariances, X[n])

    return sumOverDataPoints

