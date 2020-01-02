def IN_AutoMutualInfo(y,timeDelay = 1,estMethod = 'gaussian',extraParam = []):
    if isinstance(timeDelay,str):
        timeDelay = CO_FirstZero(y,'ac')
    N = len(y)

    if isinstance(timeDelay,list):
        numTimeDelays = len(timeDelay)
    else:
        numTimeDelays = 1
        timeDelay = [timeDelay]
    amis = []
    out = {}
    for k in range(numTimeDelays):
        y1 = y[0:N-timeDelay[k]]
        y2 = y[timeDelay[k]:N]
        if estMethod == 'gaussian':
            r = np.corrcoef(y1,y2)[1,0]
            amis.append(-.5 * np.log(1 - r**2))
            out['Auto Mutual ' + str(timeDelay[k])] = -.5 * np.log(1 - r**2)

    return out
