def SY_LocalGlobal(y,subsetHow = 'l',n = ''):

    if subsetHow == 'p' and n == '':

        n = .1

    elif n == '':

        n = 100

    N  = len(y)

    if subsetHow == 'l':

        r = range(0,min(n,N))

    elif subsetHow == 'p':

        if n > 1:

            n = .1

        r = range(0,round(N*n))

    elif subsetHow == 'unicg':

        r = np.round(np.arange(0,N,n)).astype(int)

    elif subsetHow == 'randcg':

        r = np.random.randint(N,size = n)

    if len(r)<5:

        out = np.nan
        return out

    out = {}

    out['absmean'] = np.absolute(np.mean(y[r]))
    out['std'] = np.std(y[r],ddof = 1)
    out['median'] = np.median(y[r])
    out['iqr'] = np.absolute(( 1 - stats.iqr(y[r]) /stats.iqr(y) ))

    if stats.skew(y) == 0:

        out['skew'] = np.nan

    else:

        out['skew'] = np.absolute( 1 - stats.skew(y[r]) / stats.skew(y) )

    out['kurtosis'] = np.absolute(1 - stats.kurtosis(y[r],fisher=False) / stats.kurtosis(y,fisher=False))

    out['ac1'] = np.absolute( 1- CO_AutoCorr(y[r],1,'Fourier') / CO_AutoCorr(y,1,'Fourier'))


    return out
