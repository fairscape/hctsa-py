def CO_trev(y,tau = 'ac'):

    if tau == 'ac':

        tau = CO_FirstZero(y,'ac')

    else:

        tau = CO_FirstMin(y,'mi')

    N = len(y)

    yn = y[0:N-tau]
    yn1 = y[tau:N]

    try:
        raw = np.mean(np.power(yn1-yn,3)) / np.mean(np.power(yn1 - yn,2))**(3/2)

    except:

        return({'raw':np.nan,'abs':np.nan,'num':np.nan,
                'absnum':np.nan,'denom':np.nan})

    outDict = {}

    outDict['raw'] = raw

    outDict['abs'] = np.absolute(raw)

    outDict['num'] = np.mean(np.power(yn1-yn,3))

    outDict['absnum'] = np.absolute(outDict['num'])

    outDict['denom'] = np.mean(np.power(yn1-yn,2))**(3/2)

    return outDict
