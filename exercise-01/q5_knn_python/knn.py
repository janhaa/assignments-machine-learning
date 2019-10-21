import numpy as np

def evaluatePDEAt(position, samples, k):
    distances = np.fromfunction(lambda i: np.absolute(position-samples[i]), samples.shape, dtype=int)
    distances = np.sort(distances)
    return k/(len(samples)*np.pi*distances[k]**2)


def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]
    
    # positions evenly spaced
    pos = np.arange(-5, 5.0, 0.1)

    # create empty vector for results
    results = np.empty([1, 100])

    # evaluate probability density estimate at every given point
    for j in range(results.shape[1]):
        results[0][j] = evaluatePDEAt(pos[j], samples, k)

    # shape positions into vector
    pos = np.reshape(pos, (1, 100))

    # stack vectors into matrix
    estDensity = np.stack((pos, results), axis=2)

    return estDensity[0]
