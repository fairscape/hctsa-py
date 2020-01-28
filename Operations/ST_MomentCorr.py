def ST_MomentCorr(x,windowLength = .02,wOverlap = .2,mom1 = 'mean',mom2 = 'std',whatTransform = 'none'):

    N = len(x)

    if windowLength < 1:

        windowLength = math.ceil(N*windowLength)

    if wOverlap < 1:

        wOverlap = math.floor(windowLength * wOverlap)

    if whatTransform == 'abs':

        x = np.abs(x)

    elif whatTransform == 'sq':

        x = np.sqrt(x)

    elif whatTransform == 'none':

        pass

    x_buff = BF_buffer(x,windowLength,wOverlap)

    numWindows = (N / (windowLength - wOverlap))

    if x_buff.shape[1] > numWindows:

        x_buff = x_buff[:,0:x_buff.shape[1]-1]

    pointsPerWindow = x_buff.shape[0]

    if pointsPerWindow == 1:

        return None




    M1 = SUB_calcmemoments(x_buff,mom1)
    M2 = SUB_calcmemoments(x_buff,mom2)

    R = np.corrcoef(M1,M2)

    outDict = {}

    outDict['R'] = R[1,0]

    outDict['absR'] = abs(R[1,0])

    outDict['density'] = (np.max(M1) - np.min(M1))*(np.max(M2) - np.min(M2))/N



    return outDict

def SUB_calcmemoments(x_buff,momType):

    if momType == 'mean':

        moms = np.mean(x_buff,axis = 0)

    elif momType == 'std':

        moms = np.std(x_buff,axis = 0,ddof = 1)

    elif momType == 'median':

        moms = np.median(x_buff,axis = 0)

    elif momType == 'iqr':

        moms = stats.iqr(x_buff,axis = 0)

    return moms
