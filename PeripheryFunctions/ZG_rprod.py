def ZG_rprod(X,Y):

    n,m = X.shape

    if Y.shape[0] != n or len(Y.shape) != 1:
        print('rprod error')
        return None

    Y = Y[:,None]

    Z = np.multiply(X,np.matmul(Y,np.ones((1,m))))

    return Z
