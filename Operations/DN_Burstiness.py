#@numba.jit(nopython=True,parallel=True)
def DN_Burstiness(y):
    if y.mean() == 0:
        return np.nan
    r = np.std(y) / y.mean()
    B = ( r - 1 ) / ( r + 1 )
    return(B)
