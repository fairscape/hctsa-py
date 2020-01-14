def MF_hmm_CompareNStates(y,trainp = .6,nstater = [2,3,4]):
    """
        % MF_hmm_CompareNStates     Hidden Markov Model (HMM) fitting to a time series.
        %
        % Fits HMMs with different numbers of states, and compares the resulting
        % test-set likelihoods.
        %
        % The code relies on Zoubin Gharamani's implementation of HMMs for real-valued
        % Gassian-distributed observations, including the hmm and hmm_cl routines (
        % renamed ZG_hmm and ZG_hmm_cl here).
        % Implementation of HMMs for real-valued Gaussian observations:
        % http://www.gatsby.ucl.ac.uk/~zoubin/software.html
        % or, specifically:
        % http://www.gatsby.ucl.ac.uk/~zoubin/software/hmm.tar.gz
        %
        %---INPUTS:
        %
        % y, the input time series
        %
        % trainp, the initial proportion of the time series to train the model on
        %
        % nstater, the vector of state numbers to compare. E.g., (2:4) compares a number
        %               of states 2, 3, and 4.
        %
        %---OUTPUTS: statistics on how the log likelihood of the test data changes with
        % the number of states n_{states}$. We implement the code for p_{train} = 0.6$
        % as n_{states}$ varies across the range n_{states} = 2, 3, 4$.
        % Gaussian Observation Hidden Markov Model

    """

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

        numStates = nstater[j]

        Mu, Cov, P, Pi, LL = ZG_hmm(ytrain,Ntrain,numStates,30)

        LLtrains[j] = LL[-1] / Ntrain

        lik,likv = ZG_hmm_cl(ytest,Ntest,numStates,Mu,Cov,P,Pi)

        LLtests[j] = lik / Ntest

    outDict = {}

    outDict['meanLLtrain'] = np.mean(LLtrains)
    outDict['meanLLtest'] = np.mean(LLtests)
    outDict['maxLtrain'] = np.max(LLtrains)
    outDict['maxLLtest'] = np.max(LLtests)
    outDict['meandiffLLtt'] = np.mean(np.absolute(LLtests - LLtrains))

    for i in range(Nstate - 1):

        outDict['LLtestdiff' + str(i)] = LLtests[i+1] - LLtests[i]

    return outDict
