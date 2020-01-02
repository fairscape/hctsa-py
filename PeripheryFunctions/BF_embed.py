def BF_embed(y,tau = 1,m = 2,makeSignal = 0,randomSeed = [],beVocal = 0):

    N = len(y)

    N_embed = N - (m - 1)*tau

    if N_embed <= 0:
        raise Exception('Time Series (N = %u) too short to embed with these embedding parameters')
    y_embed = np.zeros((N_embed,m))

    for i in range(1,m+1):

        y_embed[:,i-1] = y[(i-1)*tau:N_embed+(i-1)*tau]
    return(y_embed)
