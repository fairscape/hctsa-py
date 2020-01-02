def CO_glscf(y,alpha = 1.0,beta = 1.0,tau = ''):
    if tau == '':
        tau = CO_FirstZero(y,'ac')
    N = len(y)
    beta = float(beta)
    alpha = float(alpha)
    y1 = np.absolute(y[0:N-tau])
    y2 = np.absolute(y[tau:N])
    top = np.mean(np.multiply(np.power(y1,alpha),np.power(y2,beta))) - np.mean(np.power(y1,alpha)) * np.mean(np.power(y2,beta))
    bot =  np.sqrt(np.mean(np.power(y1,2*alpha)) - np.mean(np.power(y1,alpha))**2) * np.sqrt(np.mean(np.power(y2,2*beta)) - np.mean(np.power(y2,beta))**2)
    if bot == 0:
        return np.inf
    glscf = top / bot
    return glscf
