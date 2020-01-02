#@numba.jit(nopython=True)
#Quantile function seems to be slower with numba
def DN_Quantile(y,q = .5):
    # if not isinstance(y,np.ndarray):
    #     y = np.asarray(y)
    return(np.quantile(y,q))
