def ZG_hmm_cl(X,T,K,Mu,Cov,P,Pi):

    p = 1

    N = len(X)

    tiny = np.exp(-700)

    if N % T != 0:

        print('N not % T')

        return None

    N = int(N / T)

    alpha = np.zeros((T,K))
    B = np.zeros((T,K))

    k1 = (2*math.pi)**(-p/2)

    Scale = np.zeros((1,T))

    likv = np.zeros((1,N))

    for n in range(int(N)):

        B = np.zeros((T,K))
        iCov = 1 / Cov

        k2 = k1 / math.sqrt(Cov)

        for i in range(T):

            for l in range(K):

                d = Mu[l] - X[(n-1)*T+i]
                B[i,l] = k2 * np.exp(-.5 * d * iCov * d)

        scale = np.zeros((T,1))
        alpha[0,:] = np.multiply(Pi.flatten('F'),B[0,:])
        scale[0] = np.sum(alpha[0,:])
        alpha[0,:] = alpha[0,:] / scale[0]

        for i in range(1,T):

            alpha[i,:] = np.multiply( np.matmul(alpha[i-1,:] , P), B[i,:])
            scale[i] = np.sum(alpha[i,:])
            alpha[i,:] = alpha[i,:] / (scale[i] + tiny)

        likv[n] = np.sum(np.log(scale + (scale == 0) * tiny))
        Scale = Scale + np.log(scale + (scale == 0) * tiny)

    lik = np.sum(Scale)

    return lik,likv
