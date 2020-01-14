def ZG_rsum(X):

    Z = np.zeros(X.shape[0])

    for i in range(X.shape[1]):
        Z = Z + X[:,i]

    return Z
