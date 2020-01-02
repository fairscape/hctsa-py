#@numba.jit(nopython=True,parallel=True)
def DN_CustomSkewness(y,whatSkew = 'pearson'):
    if whatSkew == 'pearson':
        if np.std(y) != 0:
            return (3*np.mean(y) - np.median(y)) / np.std(y)
        else:
            return 0
    elif whatSkew == 'bowley':
        qs = np.quantile(y,[.25,.5,.75])
        if np.std(y) != 0:
            return (qs[2] + qs[0] - 2*qs[1]) / (qs[2] - qs[0])
        else:
            return 0

    else:
         raise Exception('whatSkew must be either pearson or bowley.')
