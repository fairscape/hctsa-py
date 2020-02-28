def DN_RemovePoints(y,removeHow = 'absfar',p = .1):

    if removeHow == 'absclose' or 'absclose' in removeHow:

        i =  np.argsort(-np.absolute(y),kind = 'mergesort')

    elif removeHow == 'absfar' or 'absfar' in removeHow:

        i = np.argsort(np.absolute(y),kind = 'mergesort')

    elif removeHow == 'min' or 'min' in removeHow:

        i =  np.argsort(-y,kind = 'mergesort')

    elif removeHow == 'max' or 'max' in removeHow:

        i = np.argsort(y,kind = 'mergesort')

    else:

        return

    N = len(y)

    out = {}

    rKeep = np.sort(i[0:int(np.round(N*(1-p)))])
    y_trim = y[rKeep]

    #print(rKeep)


    acf_y = SUB_acf(y,8)
    acf_y_trim = SUB_acf(y_trim,8)

    out['fzcacrat'] = CO_FirstZero(y_trim,'ac')/CO_FirstZero(y,'ac')

    out['ac1rat'] = acf_y_trim[0]/acf_y[0]

    out['ac1diff'] = np.absolute(acf_y_trim[0]-acf_y[0])

    out['ac2rat'] = acf_y_trim[1]/acf_y[1]

    out['ac2diff'] = np.absolute(acf_y_trim[1]-acf_y[1])

    out['ac3rat'] = acf_y_trim[2]/acf_y[2]

    out['ac3diff'] = np.absolute(acf_y_trim[2]-acf_y[2])

    out['sumabsacfdiff'] = sum(np.absolute(acf_y_trim-acf_y))

    out['mean'] = np.mean(y_trim)

    out['median'] = np.median(y_trim)

    out['std'] = np.std(y_trim,ddof = 1)

    if stats.skew(y) != 0:

        out['skewnessrat'] = stats.skew(y_trim)/stats.skew(y)

    else:

        out['skewnessrat'] = np.nan

    try:

        out['kurtosisrat'] = stats.kurtosis(y_trim,fisher=False)/stats.kurtosis(y,fisher=False)

    except:

        out['kurtosisrat'] = np.nan

    return out

def SUB_acf(x,n):

    acf = np.zeros(n)

    for i in range(n):

        acf[i] = CO_AutoCorr(x,i,'Fourier')

    return acf
