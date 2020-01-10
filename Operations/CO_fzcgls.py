def CO_fzcglscf(y,alpha,beta,maxtau = 'empty'):

    N = len(y)

    if maxtau == 'empty':

        maxtau = N

    glscfs = np.zeros(maxtau)

    for i in range(maxtau - 1):

        tau = i + 1

        glscfs[i] = CO_glscf(y,alpha,beta,tau)

        if i > 0 and glscfs[i] * glscfs[i-1] < 0:

            out = i  + glscfs[i]  / (glscfs[i] - glscfs[i - 1])

            return out

    return maxtau
