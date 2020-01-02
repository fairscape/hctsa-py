from scipy import stats
def DN_Mode(y):
    #y must be numpy array
    if not isinstance(y,np.ndarray):
        y = np.asarray(y)
    return float(stats.mode(y).mode)
