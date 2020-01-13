def EN_wentropy(y,whaten = 'shannon',p = ''):

    N = len(y)

    if whaten == 'shannon':

        out = wentropy(y,'shannon') / N

    elif whaten == 'logenergy':

        out = wentropy(y,'logenergy') / N

    elif whaten == 'threshold':

        if p == '':

            p = np.mean(y)

        out = wentropy(y,'threshold',p) / N

    elif whaten == 'sure':

        if p == '':

            p = np.mean(y)

        out = wentropy(y,'sure',p) / N

    elif whaten == 'norm':
        if p == '':

            p = 2

        out = wentropy(y,'norm',p) / N

    else:

        out = None
        
    return out
