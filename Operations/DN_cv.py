#import warnings
#@numba.jit(nopython=True,parallel=True)
def DN_cv(x,k = 1):
    # if k % 1 != 0 or k < 0:
    #     warnings.warn("k should probably be positive int")
    if np.mean(x) == 0:
        return np.nan
    return (np.std(x)**k) / (np.mean(x)**k)
