import numpy as np 

def evaluatePDEAt(position, samples, width):
    # 1-dimensional
    D = 1
    # normalization factor is constant so calculate it already
    normalizationFactor = 1/((2*np.pi)**(D/2)*width)
    # create array of values for the inner sum
    entries = np.fromfunction(lambda i: np.exp(-((np.absolute(position-samples[i])**2)/(2*width**2))), samples.shape, dtype=int)
    # normalize and scale
    return 1/len(samples)*normalizationFactor*np.sum(entries)

def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    # create values of points of where to evaluate at
    pos = np.arange(-5, 5.0, 0.1)

    # create empty vector for results
    results = np.empty([1, 100])

    # evaluate probability density estimate at every given point
    for j in range(results.shape[1]):
        results[0][j] = evaluatePDEAt(pos[j], samples, h)

    # shape positions into vector
    pos = np.reshape(pos, (1, 100))

    # stack vectors into matrix
    estDensity = np.stack((pos, results), axis=2)

    print(estDensity.shape)

    return estDensity[0]
