#@numba.jit(nopython=True,parallel=True)
def DN_Spread(y,spreadMeasure = 'std'):
    if spreadMeasure == 'std':
        return np.std(y)
    elif spreadMeasure == 'iqr':
        return stats.iqr(y)
    elif spreadMeasure == 'mad':
        return mad(y)
    elif spreadMeasure == 'mead':
        return mead(y)#stats.median_absolute_deviation(y)
    else:
        raise Exception('spreadMeasure must be one of std, iqr, mad or mead')
def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

def mead(data, axis=None):
    return np.median(np.absolute(data - np.median(data, axis)), axis)
