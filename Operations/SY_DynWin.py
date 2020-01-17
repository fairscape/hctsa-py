def SY_DynWin(y,maxnseg = 10):

    nsegr = np.arange(2,maxnseg + 1)

    nmov = 1

    numFeatures = 9

    fs = np.zeros((len(nsegr),numFeatures))

    taug = CO_FirstZero(y,'ac')

    for i in range(len(nsegr)):

        nseg = nsegr[i]

        wlen = math.floor( len(y) / nseg )

        inc = math.floor( wlen / nmov )

        if inc == 0:

            inc = 1

        numSteps = math.floor((len(y) - wlen) / inc) + 1

        qs = np.zeros((numSteps,numFeatures))

        for j in range(numSteps):

            ysub = y[(j)*inc:(j)*inc + wlen]

            taul = CO_FirstZero(ysub,'ac')

            qs[j,0] = np.mean(ysub)

            qs[j,1] = np.std(ysub,ddof = 1)

            qs[j,2] = stats.skew(ysub)

            qs[j,3] = stats.kurtosis(ysub)

            qs[j,4] = CO_AutoCorr(ysub,1,'Fourier')

            qs[j,5] = CO_AutoCorr(ysub,2,'Fourier')

            qs[j,6] = CO_AutoCorr(ysub,taug,'Fourier')

            qs[j,7] = CO_AutoCorr(ysub,taul,'Fourier')

            qs[j,8]  = taul

        fs[i,:] = np.std(qs,axis = 0,ddof = 1)



    fs = np.std(fs,axis  = 0,ddof = 1)


    outDict = {}

    outDict['stdmean'] = fs[0]
    outDict['stdstd'] = fs[1]
    outDict['stdskew'] = fs[2]
    outDict['stdkurt'] = fs[3]
    outDict['stdac1'] = fs[4]
    outDict['stdac2'] = fs[5]
    outDict['stdactaug'] = fs[6]
    outDict['stdactaul'] = fs[7]
    outDict['stdtaul'] = fs[8]


    return outDict
