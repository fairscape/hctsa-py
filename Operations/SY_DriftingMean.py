def SY_DriftingMean(y,howl = 'num',l = ''):

    N = len(y)

    if howl == 'num':

        if l != '':

            l = math.floor(N/l)

    if l == '':

        if howl == 'num':

            l = 5

        elif howl == 'fix':

            l = 200

    if l == 0 or N < l:

        return

    numFits = math.floor(N / l)
    z = np.zeros((l,numFits))

    for i in range(0,numFits):

        z[:,i] = y[i*l :(i + 1)*l]



    zm = np.mean(z,axis = 0)
    zv = np.var(z,axis = 0,ddof = 1)

    meanvar = np.mean(zv)
    maxmean = np.max(zm)
    minmean = np.min(zm)
    meanmean = np.mean(zm)

    outDict = {}

    outDict['max'] = maxmean/meanvar
    outDict['min'] = minmean/meanvar
    outDict['mean'] = meanmean/meanvar
    outDict['meanmaxmin'] = (outDict['max']+outDict['min'])/2
    outDict['meanabsmaxmin'] = (np.absolute(outDict['max'])+np.absolute(outDict['min']))/2


    return outDict
