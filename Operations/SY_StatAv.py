def SY_StatAv(y,whatType = 'seg',n = 5):

    N = len(y)

    if whatType == 'seg':

        M = np.zeros(n)
        p = math.floor(N/n)

        for j in range(1,n+1):

            M[j - 1] = np.mean(y[p*(j-1):p*j])

    elif whatType == 'len':

        if N > 2*n:

            pn = math.floor(N / n)
            M = np.zeros(pn)

            for j in range(1,pn + 1):

                M[j-1] = np.mean(y[(j-1)*n:j*n])

        else:

            return

    s = np.std(y,ddof = 1)
    sdav = np.std(M,ddof = 1)

    out = sdav / s

    return out
