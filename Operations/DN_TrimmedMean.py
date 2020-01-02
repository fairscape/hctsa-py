#@numba.jit(nopython=True,parallel=True)
def DN_TrimmedMean(y,n = 0):
    N = len(y)
    trim = int(np.round(N * n / 2))
    y = np.sort(y)
    #return stats.trim_mean(y,n) doesn't agree with matlab
    return np.mean(y[trim:N-trim])
