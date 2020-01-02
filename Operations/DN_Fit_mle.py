def DN_Fit_mle(y,fitWhat = 'gaussian'):
    if fitWhat == 'gaussian':
        phat = stats.norm.fit(y)
        out = {'mean':phat[0],'std':phat[1]}
        return out
    else:
        print('Use gaussian geometric not implemented yet')
