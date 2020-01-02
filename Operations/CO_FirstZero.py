def CO_FirstZero(y,corrFun = 'ac'):
    acf = CO_AutoCorr(y,[],'Fourier')
    N = len(y)
    for i in range(1,N-1):
        if acf[i] < 0:
            return i
    return N
