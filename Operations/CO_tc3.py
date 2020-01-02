def CO_tc3(y,tau = 'ac'):
    if tau == 'ac':
        tau = CO_FirstZero(y,'ac')
    else:
        tau = CO_FirstMin(y,'mi')
    N = len(y)
    yn = y[0:N-2*tau]
    yn1 = y[tau:N-tau]
    yn2 = y[tau*2:N]
    raw = np.mean(np.multiply(np.multiply(yn,yn1),yn2)) / (np.absolute(np.mean(np.multiply(yn,yn1))) ** (3/2))

    return raw
