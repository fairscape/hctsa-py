def CO_NonlinearAutocorr(y,taus,doAbs ='empty'):

    if doAbs == 'empty':

        if len(taus) % 2 == 1:

            doAbs = 0

        else:

            doAbs = 1

    N = len(y)
    tmax = np.max(taus)

    nlac = y[tmax:N]

    for i in taus:

        nlac = np.multiply(nlac,y[ tmax - i:N - i ])

    if doAbs:

        return np.mean(np.absolute(nlac))

    else:

        return np.mean(nlac)
