def DN_OutlierTest(y,p = 2,justMe=''):

    outDict = {}

    index = np.logical_and(y > np.percentile(y,p),y < np.percentile(y,100-p))

    outDict['mean'] = np.mean(y[index])

    outDict['std'] = np.std(y[index]) / np.std(y)

    if justMe == 'mean':

        return outDict['mean']

    elif justMe == 'std':

        return outDict['std']

    return outDict
