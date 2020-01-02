#@numba.jit(nopython=True,parallel=True)
def EN_ApEn(y,mnom = 1,rth = .2):

    r = rth * np.std(y)
    N = len(y)
    phi = np.zeros(2)

    for k in range(2):
        m = mnom + k
        m = int(m)
        C = np.zeros(N-m+1)

        x = np.zeros((N - m + 1, m))

        for i in range(N - m + 1):
            x[i,:] = y[i:i+m]

        ax = np.ones((N - m + 1, m))
        for i in range(N-m+1):

            for j in range(m):
                ax[:,j] = x[i,j]

            d = np.absolute(x-ax)
            if m > 1:
                d = np.maximum(d[:,0],d[:,1])
            dr = ( d <= r )
            C[i] = np.sum(dr) / (N-m+1)
        phi[k] = np.mean(np.log(C))
    return phi[0] - phi[1]
