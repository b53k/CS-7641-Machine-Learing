import numpy as np


def create_nl_feature(X):
    '''
    TODO - Create additional features and add it to the dataset
    
    returns:
        X_new - (N, d + num_new_features) array with 
                additional features added to X such that it
                can classify the points in the dataset.
    '''
    
    #raise NotImplementedError
    squares = X[:,:]**2
    cubes = X[:,:]**3

    N = X.shape[0]
    features = np.zeros((N,1))
    for idx in range(X.shape[0]):
        features[idx] = X[idx][0] * X[idx][1]  # x1 * x2 and so on.
    
    X_new = np.concatenate((X,features), axis = 1)
    #X_new = np.concatenate((X,features, squares, cubes), axis = 1)

    return X_new