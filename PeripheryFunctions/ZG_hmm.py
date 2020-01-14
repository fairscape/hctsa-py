def ZG_hmm(X, T = '', K = 2, cyc = 100, tol = .0001):
    """
    Python Implementation currently slow used for loops same as orginal
    need to vectorize
    %
    % X - N x p data matrix
    % T - length of each sequence (N must evenly divide by T, default T=N)
    % K - number of states (default 2)
    % cyc - maximum number of cycles of Baum-Welch (default 100)
    % tol - termination tolerance (prop change in likelihood) (default 0.0001)
    %
    % Mu - mean vectors
    % Cov - output covariance matrix (full, tied across states)
    % P - state transition matrix
    % Pi - priors
    % LL - log likelihood curve
    %
    % Iterates until a proportional change < tol in the log likelihood
    % or cyc steps of Baum-Welch
    %
    % Machine Learning Toolbox
    % Version 1.0  01-Apr-96
    % Copyright (c) by Zoubin Ghahramani
    % http://mlg.eng.cam.ac.uk/zoubin/software.html
    %
    % ------------------------------------------------------------------------------
    % The MIT License (MIT)
    %
    % Copyright (c) 1996, Zoubin Ghahramani
    %
    % Permission is hereby granted, free of charge, to any person obtaining a copy
    % of this software and associated documentation files (the "Software"), to deal
    % in the Software without restriction, including without limitation the rights
    % to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    % copies of the Software, and to permit persons to whom the Software is
    % furnished to do so, subject to the following conditions:
    %
    % The above copyright notice and this permission notice shall be included in
    % all copies or substantial portions of the Software.
    %
    % THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    % IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    % FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    % AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    % LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    % OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    % THE SOFTWARE.
    % ------------------------------------------------------------------------------

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

    LL = []
    lik = 0

    alpha = np.zeros((T,K))
    beta = np.zeros((T,K))
    gamma = np.zeros((T,K))

    B = np.zeros((T,K))
    k1 = (2*math.pi)**(-p/2)

    for cycle in range(cyc):

        Gamma = []
        Gammasum = np.zeros((1,K))
        Scale = np.zeros((T,1))
        Xi = np.zeros((T-1,K*K))

        for n in range(int(N)):

            #Assuming P = 1 makes Cov single value not matrix
            iCov = 1 / Cov



            k2 = k1 / np.sqrt(Cov)

            #get rid of for loops
            for i in range(T):

                for l in range(K):

                    d = Mu[l] - X[(n-1)*T + i]

                    B[i,l] = k2*np.exp(-.5*d*iCov*d)

            scale = np.zeros((T,1))
            alpha[0,:] = np.multiply(Pi,B[0,:])
            scale[0] = np.sum(alpha[0,:])
            alpha[0,:] = alpha[0,:] / scale[0]

            for i in range(1,T):

                alpha[i,:] = np.multiply( np.matmul(alpha[i-1,:] , P), B[i,:])
                scale[i] = np.sum(alpha[i,:])
                alpha[i,:] = alpha[i,:] / scale[i]

            beta[T - 1,:] = np.ones((1,K)) / scale[T-1]

            for i in range(T-2,-1,-1):

                beta[i,:] = np.matmul(np.multiply(beta[i+1,:],B[i+1,:]),P.T) / scale[i]

            gamma = np.multiply(alpha,beta)
            gamma = ZG_rdiv(gamma,ZG_rsum(gamma))
            gammasum = np.sum(gamma,axis = 0)

            xi = np.zeros((T-1,K*K))

            for i in range(T-1):

                t = np.multiply(P,np.matmul(alpha[i,:].T,np.multiply(beta[i+1,:],B[i+1,:])))
                xi[i,:] = t.flatten('F') / np.sum(t)

            Scale = Scale + np.log(scale)

            if Gamma == []:

                Gamma = gamma

            else:

                Gamma = np.vstack((Gamma,gamma))

            Gammasum = Gammasum + gammasum
            Xi = Xi + xi

        Mu = np.zeros((K,p))
        Mu = np.matmul(Gamma.T,X)

        Mu = ZG_rdiv(Mu,Gammasum.T)


        sxi = np.transpose(ZG_rsum(Xi.T))
        sxi = np.reshape(sxi,(K,K)).T

        P = ZG_rdiv(sxi,ZG_rsum(sxi))

        Pi = np.zeros((1,K))

        #can vectorize below
        for i in range(int(N)):

            Pi = Pi + Gamma[(i-1)*T + 1,:]

        Pi = Pi / N

        Cov = 0



        for l in range(K):

            d = X-Mu[l]
            Cov = Cov + np.matmul(ZG_rprod(d,Gamma[:,l]).T,d)

        Cov = Cov / np.sum(Gammasum)

        oldlik = lik
        lik = np.sum(Scale)
        LL.append(lik)


        if cycle <= 1:

            likbase = lik


        elif lik < oldlik:

            print("Error old lik better")

        elif ((lik-likbase)<(1 + tol)*(oldlik-likbase)):
            break

    return Mu, Cov, P, Pi, LL
