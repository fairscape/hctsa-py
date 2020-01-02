def DN_nlogL_norm(y):
    muhat, sigmahat = stats.norm.fit(y)
    z = (y - muhat) / sigmahat
    L = -.5*np.power(z,2) - np.log(np.sqrt(2*math.pi)*sigmahat)
    return -sum(L) / len(y) 
