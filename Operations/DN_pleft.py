#@numba.jit(nopython=True,parallel=True)
#oddly this function slows down with numba
def DN_pleft(y,th = .1):

    p  = np.quantile(np.absolute(y - np.mean(y)),1-th)


    return p / np.std(y)
