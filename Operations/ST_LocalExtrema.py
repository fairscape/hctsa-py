def ST_LocalExtrema(y,lorf = 'l',n = ''):
    if lorf == 'l' and n == '':
        n = 100
    elif n == '':
        n = 5

    N = len(y)

    if lorf == 'l':
        wl = n
    elif lorf == 'n':
        wl = math.floor(N/n)
    else:
        wl = CO_FirstZero(y,'ac')

    if wl > N or wl <= 1:
        #print('window too short or long')
        return np.nan

    y_buffer = BF_makeBuffer(y,wl).transpose()

    numWindows = y_buffer.shape[1]

    locmax = np.max(y_buffer,axis = 0)

    locmin = np.min(y_buffer,axis = 0)

    abslocmin = np.absolute(locmin)

    exti = np.where(abslocmin > locmax)

    locext = locmax

    locext[exti] = locmin[exti]

    abslocext = np.absolute(locext)

    out = {}

    out['meanrat'] = np.mean(locmax)/np.mean(abslocmin)
    out['medianrat'] = np.median(locmax)/np.median(abslocmin)
    out['minmax'] = np.min(locmax)
    out['minabsmin'] = np.min(abslocmin)
    out['minmaxonminabsmin'] = np.min(locmax)/np.min(abslocmin)
    out['meanmax'] = np.mean(locmax)
    out['meanabsmin'] = np.mean(abslocmin)
    out['meanext'] = np.mean(locext)
    out['medianmax'] = np.median(locmax)
    out['medianabsmin'] = np.median(abslocmin)
    out['medianext'] = np.median(locext)
    out['stdmax'] = np.std(locmax,ddof=1)
    out['stdmin'] = np.std(locmin,ddof=1)
    out['stdext'] = np.std(locext,ddof=1)
    #out.zcext = ST_SimpleStats(locext,'zcross');
    out['meanabsext'] = np.mean(abslocext)
    out['medianabsext'] = np.median(abslocext)
    out['diffmaxabsmin'] = np.sum(np.absolute(locmax-abslocmin))/numWindows
    out['uord'] = np.sum(np.sign(locext))/numWindows #% whether extreme events are more up or down
    out['maxmaxmed'] = np.max(locmax)/np.median(locmax)
    out['minminmed'] = np.min(locmin)/np.median(locmin)
    out['maxabsext'] = np.max(abslocext)/np.median(abslocext)

    return out
