import scipy

def SB_TransitionMatrix(y,howtocg = 'quantile',numGroups = 2,tau = 1):

    if tau == 'ac':
        tau = CO_FirstZero(y,'ac')

    if tau > 1:

        y = scipy.signal.resample(y,math.ceil(len(y) / tau))

    N = len(y)

    yth = SB_CoarseGrain(y,howtocg,numGroups)

    if yth.shape[1] > yth.shape[0]:

        yth = yth.transpose()

    T = np.zeros((numGroups,numGroups))


    for i in range(0,numGroups):

        ri = (yth == i + 1)

        if sum(ri) == 0:

            T[i,:] = 0

        else:

            ri_next = np.append([False],ri[:-1])

            for j in range(numGroups):

                T[i,j] = np.sum((yth[ri_next] == j + 1))
    out = {}

    T = T / ( N - 1 )

    if numGroups == 2:

        for i in range(2):

            for j in range(2):

                out['T' + str(i) + str(j)] = T[i,j]

    elif numGroups == 3:

        for i in range(3):

            for j in range(3):

                out['T' + str(i) + str(j)] = T[i,j]

    elif numGroups > 3:

        for i in range(numGroups):

            out['TD' + str(i)] = T[i,i]


    out['ondiag'] = np.sum(np.diag(T))

    out['stddiag'] = np.std(np.diag(T))

    out['symdiff'] = np.sum(np.sum(np.absolute(T-T.transpose())))

    out['symsumdiff'] = np.sum(np.sum(np.tril(T,-1)) - np.sum(np.triu(T,1)))

    covT = np.cov(T.transpose())


    out['sumdiagcov'] = np.sum(np.diag(covT))

    eigT = np.linalg.eigvals(T)

    out['stdeig'] = np.std(eigT)

    out['maxeig'] = np.real(np.max(eigT))

    out['mineig'] = np.real(np.min(eigT))

    eigcovT = np.linalg.eigvals(covT)

    out['stdcoveig'] = np.std(eigcovT)

    out['maxcoveig'] = np.max(eigcovT)

    out['mincoveig'] = np.min(eigcovT)

    return out
