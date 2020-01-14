def ZG_rdiv(X,Y):

    if X.shape[0] != Y.shape[0] or len(Y.shape) != 1:
        if len(Y.shape) > 1:
            if Y.shape[1] != 1:
                    print('Error')
                    return None
        else:
            print('Error')
            return None

    Z = np.zeros(X.shape)

    if len(X.shape) == 1:

        for i in range(X.shape[0]):
            Z[i] = X[i] / Y[i]
        
        return Z

    for i in range(X.shape[1]):

        Z[:,i] = np.divide(X[:,i],Y.flatten())

    return Z
