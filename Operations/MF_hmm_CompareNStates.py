def MF_hmm_CompareNStates(y,trainp = .6,nstater = [2,3,4]):

    N = len(y)

    Ntrain = math.floor(N * trainp)

    ytrain = y[:Ntrain]

    if Ntrain < N:

        ytest = y[Ntrain:]
        Ntest = len(ytest)

    Nstate = len(nstater)
    LLtrains = np.zeros(Nstate)
    LLtests = np.zeros(Nstate)

    for j in range(Nstate):

        numStates = nstater[i]

        Mu, Cov, P, Pi, LL = ZG_hmm(ytrain,Ntrain,numStates,30)
