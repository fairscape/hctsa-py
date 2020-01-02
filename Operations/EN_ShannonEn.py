#@numba.jit(nopython=True,parallel=True)
def EN_ShannonEn(y):
    p = np.zeros(len(np.unique(y)))
    n = 0
    for i in np.unique(y):
        p[n] = len(y[y == i]) / len(y)
        n = n + 1

    return -np.sum(p*np.log2(p))
