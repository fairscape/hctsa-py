def SY_SpreadRandomLocal(y,l = 100,numSegs = 25,randomSeed = 0):

    if isinstance(l,str):
        taug = CO_FirstZero(y,'ac')

        if l == 'ac2':
            l = 2*taug
        else:
            l = 5*taug

    N = len(y)

    if l > .9 * N:
        #print('Time series too short for given l')
        return np.nan

    numFeat = 8

    qs = np.zeros((numSegs,numFeat))

    for j in range(numSegs):

        ist = np.random.randint(N - l)
        ifh = ist + l

        ysub = y[ist:ifh]

        taul = CO_FirstZero(ysub,'ac')

        qs[j,0] = np.mean(ysub)

        qs[j,1] = np.std(ysub)

        qs[j,2] = stats.skew(ysub)

        qs[j,3] = stats.kurtosis(ysub)

        #entropyDict = EN_SampEn(ysub,1,.15)

        #qs[j,4] = entropyDict['Quadratic Entropy']

        qs[j,5] =  CO_AutoCorr(ysub,1,'Fourier')

        qs[j,6] = CO_AutoCorr(ysub,2,'Fourier')

        qs[j,7] = taul


    fs = np.zeros((numFeat,2))

    fs[:,0] = np.nanmean(qs,axis = 0)

    fs[:,1] = np.nanstd(qs,axis = 0)

    out = {}

    out['meanmean'] = fs[0,0]

    out['meanstd'] = fs[1,0]

    out['meanskew'] = fs[2,0]

    out['meankurt'] = fs[3,0]

    #out['meansampEn'] = fs[4,0]

    out['meanac1'] = fs[5,0]

    out['meanac2'] = fs[6,0]

    out['meantaul'] = fs[7,0]


    out['stdmean'] = fs[0,1]

    out['stdstd'] = fs[1,1]

    out['stdskew'] = fs[2,1]

    out['stdkurt'] = fs[3,1]

    #out['stdsampEn'] = fs[4,1]

    out['stdac1'] = fs[5,1]

    out['stdac2'] = fs[6,1]

    out['stdtaul'] = fs[7,1]

    return out
