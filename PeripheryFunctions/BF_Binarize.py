def BF_Binarize(y,binarizeHow = 'diff'):

    if binarizeHow == 'diff':

        yBin = stepBinary(np.diff(y))

    if binarizeHow == 'mean':

        yBin = stepBinary(y - np.mean(y))

    if binarizeHow == 'median':

        yBin = stepBinary(y - np.median(y))

    if binarizeHow == 'iqr':

        iqr = np.quantile(y,[.25,.75])

        iniqr = np.logical_and(y > iqr[0], y<iqr[1])

        yBin = np.zeros(len(y))

        yBin[iniqr] = 1

    return yBin

def stepBinary(X):

    Y = np.zeros(len(X))

    Y[ X > 0 ] = 1

    return Y
