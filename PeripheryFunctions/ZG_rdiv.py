def ZG_rdiv(X,Y):

    if X.shape[0] != Y.shape[0] or len(Y.shape) != 1:

        return None

    Z = np.zeros(X.shape)

    for i in range(X.shape[1]):

        Z[:,i] = np.divide(X[:,i],Y)

    return Z
