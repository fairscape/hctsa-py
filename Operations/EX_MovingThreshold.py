def EX_MovingThreshold(y,a = 1,b = .1):

    if b < 0 or b > 1:

        print("b must be between 0 and 1")
        return None

    N = len(y)
    y = np.absolute(y)
    q = np.zeros(N)
    kicks = np.zeros(N)

    q[0] = 1

    for i in range(1,N):

        if y[i] > q[i-1]:

            q[i] = (1 + a) * y[i]
            kicks[i] = q[i] - q[i-1]

        else:

            q[i] = ( 1 - b ) *  q[i-1]


    outDict = {}

    outDict['meanq'] = np.mean(q)
    outDict['medianq'] = np.median(q)
    outDict['iqrq'] = stats.iqr(q)
    outDict['maxq'] = np.max(q)
    outDict['minq'] = np.min(q)
    outDict['stdq'] = np.std(q)
    outDict['meanqover'] = np.mean(q - y)



    outDict['pkick'] = np.sum(kicks) / N - 1
    fkicks = np.argwhere(kicks > 0).flatten()
    Ikicks = np.diff(fkicks)
    outDict['stdkicks'] = np.std(Ikicks)
    outDict['meankickf'] = np.mean(Ikicks)
    outDict['mediankicksf'] = np.median(Ikicks)

    return outDict
