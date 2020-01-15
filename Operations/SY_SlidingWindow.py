def SY_SlidingWindow(y,windowStat = 'mean',acrossWinStat='std',numSeg=5,incMove=2):

    winLength = math.floor(len(y) / numSeg)

    if winLength == 0:

        return

    inc = math.floor(winLength / incMove)

    if inc == 0:

        inc = 1


    numSteps = (math.floor((len(y)-winLength)/inc) + 1)

    qs = np.zeros(numSteps)

    if windowStat == 'mean':

        for i in range(numSteps):

            qs[i] = np.mean(y[getWindow(i)])

    elif windowStat == 'std':

        for i in range(numSteps):

            qs[i] = np.std(y[getWindow(i)])


    elif windowStat == 'apen':

        for i in range(numSteps):

            qs[i] = EN_ApEn(y[getWindow(i)],1,.2)

    elif windowStat == 'sampen':

        for i in range(numSteps):

            sampStruct = EN_SampEn(y[getWindow(i)],1,.1)
            qs[i] = sampStruct['Sample Entropy']

    elif windowStat == 'mom3':

        for i in range(numSteps):

            qs[i] = DN_Moments(y[getWindow(i)],3)

    elif windowStat == 'mom4':

        for i in range(numSteps):

            qs[i] = DN_Moments(y[getWindow(i)],4)

    elif windowStat == 'mom5':

        for i in range(numSteps):

            qs[i] = DN_Moments(y[getWindow(i)],5)

    elif windowStat == 'AC1':

        for i in range(numSteps):

            qs[i] = CO_AutoCorr(y[getWindow(i)],1,'Fourier')


    if acrossWinStat == 'std':

        out = np.std(qs)  / np.std(y)

    elif acrossWinStat == 'apen':

        out = EN_ApEn(qs,2,.15)

    elif acrossWinStat == 'sampen':

        out = EN_SampEn(qs,2,.15)['Sample Entropy']

    else:

        out = None

    return out



def getWindow(stepInd,inc,winLength):

    return np.arange((stepInd - 1)*inc,(stepInd - 1)*inc + winLength)
