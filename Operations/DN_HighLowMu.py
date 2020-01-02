#@numba.jit(nopython=True,parallel=True)
def DN_HighLowMu(y):
    mu = np.mean(y)
    mhi = np.mean(y[y>mu])
    mlo = np.mean(y[y<mu])
    return (mhi - mu) / (mu - mlo)
