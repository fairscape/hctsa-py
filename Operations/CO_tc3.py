def CO_tc3(y,tau = 'ac'):

    if tau == 'ac':

        tau = CO_FirstZero(y,'ac')

    elif tau == 'mi':

        tau = CO_FirstMin(y,'mi')

    N = len(y)

    yn = y[0:N-2*tau]
    yn1 = y[tau:N-tau]
    yn2 = y[tau*2:N]

    try:

        raw = np.mean(np.multiply(np.multiply(yn,yn1),yn2)) / (np.absolute(np.mean(np.multiply(yn,yn1))) ** (3/2))

    except:

        return({'raw':np.nan,'abs':np.nan,'num':np.nan,
                'absnum':np.nan,'denom':np.nan})

    outDict = {}

    outDict['raw'] = raw

    outDict['abs'] = np.absolute(raw)

    outDict['num'] = np.mean(np.multiply(yn,np.multiply(yn1,yn2)))

    outDict['absnum'] = np.absolute(outDict['num'])

    outDict['denom'] = np.absolute( np.mean(np.multiply(yn,yn1)))**(3/2)

    return outDict
