def PH_ForcePotential(y,whatPotential = 'dblwell',params = []):

    if params == []:

        if whatPotential == 'dblwell':

            params = [2,.1,.1]

        elif whatPotential == 'sine':

            params = [1,1,1]

        else:

            print('Unreconginzed Potential')
            return

    N  = len(y)

    if len(params) < 3:

        print('3 Parameters required')
        return

    alpha = params[0]
    kappa = params[1]
    deltat = params[2]

    if whatPotential == 'dblwell':

        def F(x): return - np.power(x,3) + alpha**2 * x

        def V(x): return np.power(x,4) / 4 - alpha**2 * np.power(x,2) / 2

    if whatPotential == 'sine':

        def F(x): return np.sin(x / alpha) / alpha

        def V(x): return - np.cos(x / alpha)

    x = np.zeros(N)
    v = np.zeros(N)

    for i in range(1,N):

        x[i] =  x[i-1] + v[i-1]*deltat+(F(x[i-1])+y[i-1]-kappa*v[i-1])*deltat**2
        v[i] = v[i-1] + (F(x[i-1])+y[i-1]-kappa*v[i-1])*deltat

    if np.isnan(x[-1]) or np.absolute(x[-1]) > 1000000000:

        print('Trajectroy Blew out!')
        return

    outDict = {}

    outDict['mean'] = np.mean(x)
    outDict['median'] = np.median(x)
    outDict['std'] = np.std(x)
    outDict['range'] = np.max(x) - np.min(x)
    outDict['proppos'] = np.sum((x > 0)) / N
    outDict['pcross'] = np.sum((np.multiply(x[:-1],x[1:]) < 0)) / ( N - 1 )
    outDict['ac1'] = np.absolute(CO_AutoCorr(x,1,'Fourier'))
    outDict['ac10'] = np.absolute(CO_AutoCorr(x,10,'Fourier'))
    outDict['ac25'] = np.absolute(CO_AutoCorr(x,25,'Fourier'))
    outDict['tau'] = CO_FirstZero(x,'ac')
    outDict['finaldev'] = np.absolute(x[-1])

    if whatPotential == 'dblwell':

        outDict['pcrossup'] = np.sum((np.multiply(x[:-1] - alpha,x[1:]-alpha) < 0)) / (N-1)
        outDict['pcrossdown'] = np.sum((np.multiply(x[:-1] + alpha,x[1:]+alpha) < 0)) / (N-1)


    return outDict
