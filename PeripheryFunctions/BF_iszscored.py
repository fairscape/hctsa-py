def BF_iszscored(x):
    numericThreshold = 2.2204E-16
    iszscored = ((np.absolute(np.mean(x)) < numericThreshold) & (np.absolute(np.std(x)-1) < numericThreshold))
    return(iszscored)
