def DN_Moments(y,theMom = 1):
    if np.std(y) != 0:
        return stats.moment(y,theMom) / np.std(y)
    else:
        return 0
