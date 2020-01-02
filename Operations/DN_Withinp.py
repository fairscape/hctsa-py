#@numba.jit(nopython=True,parallel=True)
def DN_Withinp(x,p = 1,meanOrMedian = 'mean'):
    N = len(x)

    if meanOrMedian == 'mean':
        mu = np.mean(x)
        sig = np.std(x)
    elif meanOrMedian == 'median':
        mu = np.median(x)
        sig = 1.35*stats.iqr(x)
    else:
        raise Exception('Unknown meanOrMedian should be mean or median')
    return np.sum((x >= mu-p*sig) & (x <= mu + p*sig)) / N
