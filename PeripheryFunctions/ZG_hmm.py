def ZG_hmm(X, T = '', K = 2, cyc = 100, tol = .0001):
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
    """

    #For my purpose X should always be Nx1

    N = len(X)
    p = 1
    if T == '':

        T = N

    if N%T != 0:

        return None

    N = N / T

    Cov = np.cov(X)

    Mu = np.random.normal(0,1,K) * math.sqrt(Cov) + np.ones(K)*np.mean(X)

    Pi = np.random.normal(0,1,(1,K))
    Pi = Pi / np.sum(Pi)

    P = np.random.uniform(0,1,(K,K))
    P = ZG_rdiv(P,ZG_rsum(P))
