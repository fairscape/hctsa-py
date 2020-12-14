def DN_Moments(y,theMom = 1):
    if np.std(y) != 0:
        return stats.moment(y,theMom) / np.std(y)
    else:
        return 0

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

#@numba.jit(nopython=True,parallel=True)
def DN_Withinp(x,p = 1,meanOrMedian = 'mean'):
    N = len(x)

    if meanOrMedian == 'mean':
        mu = np.mean(x)
        sig = np.std(x)
    elif meanOrMedian == 'median':
        mu = np.median(x)
        sig = 1.35*stats.iqr(x)
    else:
        raise Exception('Unknown meanOrMedian should be mean or median')
    return np.sum((x >= mu-p*sig) & (x <= mu + p*sig)) / N

def SY_SpreadRandomLocal(y,l = 100,numSegs = 25,randomSeed = 0):

    if isinstance(l,str):
        taug = CO_FirstZero(y,'ac')

        if l == 'ac2':
            l = 2*taug
        else:
            l = 5*taug

    N = len(y)

    if l > .9 * N:
        #print('Time series too short for given l')
        return np.nan

    numFeat = 8

    qs = np.zeros((numSegs,numFeat))

    for j in range(numSegs):

        ist = np.random.randint(N - l)
        ifh = ist + l

        ysub = y[ist:ifh]

        taul = CO_FirstZero(ysub,'ac')

        qs[j,0] = np.mean(ysub)

        qs[j,1] = np.std(ysub)

        qs[j,2] = stats.skew(ysub)

        qs[j,3] = stats.kurtosis(ysub)

        #entropyDict = EN_SampEn(ysub,1,.15)

        #qs[j,4] = entropyDict['Quadratic Entropy']

        qs[j,5] =  CO_AutoCorr(ysub,1,'Fourier')

        qs[j,6] = CO_AutoCorr(ysub,2,'Fourier')

        qs[j,7] = taul


    fs = np.zeros((numFeat,2))

    fs[:,0] = np.nanmean(qs,axis = 0)

    fs[:,1] = np.nanstd(qs,axis = 0)

    out = {}

    out['meanmean'] = fs[0,0]

    out['meanstd'] = fs[1,0]

    out['meanskew'] = fs[2,0]

    out['meankurt'] = fs[3,0]

    #out['meansampEn'] = fs[4,0]

    out['meanac1'] = fs[5,0]

    out['meanac2'] = fs[6,0]

    out['meantaul'] = fs[7,0]


    out['stdmean'] = fs[0,1]

    out['stdstd'] = fs[1,1]

    out['stdskew'] = fs[2,1]

    out['stdkurt'] = fs[3,1]

    #out['stdsampEn'] = fs[4,1]

    out['stdac1'] = fs[5,1]

    out['stdac2'] = fs[6,1]

    out['stdtaul'] = fs[7,1]

    return out

#@numba.jit(nopython=True)
#Quantile function seems to be slower with numba
def DN_Quantile(y,q = .5):
    # if not isinstance(y,np.ndarray):
    #     y = np.asarray(y)
    return(np.quantile(y,q))

def SY_StatAv(y,whatType = 'seg',n = 5):

    N = len(y)

    if whatType == 'seg':

        M = np.zeros(n)
        p = math.floor(N/n)

        for j in range(1,n+1):

            M[j - 1] = np.mean(y[p*(j-1):p*j])

    elif whatType == 'len':

        if N > 2*n:

            pn = math.floor(N / n)
            M = np.zeros(pn)

            for j in range(1,pn + 1):

                M[j-1] = np.mean(y[(j-1)*n:j*n])

        else:

            return

    s = np.std(y,ddof = 1)
    sdav = np.std(M,ddof = 1)

    out = sdav / s

    return out

def DN_RemovePoints(y,removeHow = 'absfar',p = .1):

    if removeHow == 'absclose' or 'absclose' in removeHow:

        i =  np.argsort(-np.absolute(y),kind = 'mergesort')

    elif removeHow == 'absfar' or 'absfar' in removeHow:

        i = np.argsort(np.absolute(y),kind = 'mergesort')

    elif removeHow == 'min' or 'min' in removeHow:

        i =  np.argsort(-y,kind = 'mergesort')

    elif removeHow == 'max' or 'max' in removeHow:

        i = np.argsort(y,kind = 'mergesort')

    else:

        return

    N = len(y)

    out = {}

    rKeep = np.sort(i[0:int(np.round(N*(1-p)))])
    y_trim = y[rKeep]

    #print(rKeep)


    acf_y = SUB_acf(y,8)
    acf_y_trim = SUB_acf(y_trim,8)

    out['fzcacrat'] = CO_FirstZero(y_trim,'ac')/CO_FirstZero(y,'ac')

    out['ac1rat'] = acf_y_trim[0]/acf_y[0]

    out['ac1diff'] = np.absolute(acf_y_trim[0]-acf_y[0])

    out['ac2rat'] = acf_y_trim[1]/acf_y[1]

    out['ac2diff'] = np.absolute(acf_y_trim[1]-acf_y[1])

    out['ac3rat'] = acf_y_trim[2]/acf_y[2]

    out['ac3diff'] = np.absolute(acf_y_trim[2]-acf_y[2])

    out['sumabsacfdiff'] = sum(np.absolute(acf_y_trim-acf_y))

    out['mean'] = np.mean(y_trim)

    out['median'] = np.median(y_trim)

    out['std'] = np.std(y_trim,ddof = 1)

    if stats.skew(y) != 0:

        out['skewnessrat'] = stats.skew(y_trim)/stats.skew(y)
    try:

        out['kurtosisrat'] = stats.kurtosis(y_trim,fisher=False)/stats.kurtosis(y,fisher=False)

    except:

        pass

    return out

def SUB_acf(x,n):

    acf = np.zeros(n)

    for i in range(n):

        acf[i] = CO_AutoCorr(x,i,'Fourier')

    return acf

def DK_lagembed(x,M,lag = 1):

    advance = 0

    #Should make sure x is column
    #for me pretty much always just array doesnt matter

    lx = len(x)

    newsize = lx - lag*(M-1)

    y = np.zeros((newsize,M))

    i = 0

    for j in range(0,lag*(-(M)),-lag):

        first =  lag*(M-1) + j

        last = first + newsize

        if last > lx:

            last = lx - 1

        y[:,i] = x[first:last]

        i = i + 1

    return y

def CO_NonlinearAutocorr(y,taus,doAbs ='empty'):

    if doAbs == 'empty':

        if len(taus) % 2 == 1:

            doAbs = 0

        else:

            doAbs = 1

    N = len(y)
    tmax = np.max(taus)

    nlac = y[tmax:N]

    for i in taus:

        nlac = np.multiply(nlac,y[ tmax - i:N - i ])

    if doAbs:

        return np.mean(np.absolute(nlac))

    else:

        return np.mean(nlac)

def DN_OutlierInclude(y,thresholdHow='abs',inc=.01):
    if not BF_iszscored(y):
        muhat, sigmahat = stats.norm.fit(y)
        y = (y - muhat) / sigmahat
        #warnings.warn('DN_OutlierInclude y should be z scored. So just converted y to z-scores')
    N = len(y)
    if thresholdHow == 'abs':
        thr = np.arange(0,np.max(np.absolute(y)),inc)
        tot = N
    if thresholdHow == 'p':
        thr = np.arange(0,np.max(y),inc)
        tot = sum( y >= 0)
    if thresholdHow == 'n':
        thr = np.arange(0,np.max(-y),inc)
        tot = sum( y <= 0)
    msDt = np.zeros((len(thr),6))
    for i in range(len(thr)):
        th = thr[i]

        if thresholdHow == 'abs':
            r = np.where(np.absolute(y) >= th)
        if thresholdHow == 'p':
            r = np.where(y >= th)
        if thresholdHow == 'n':
            r = np.where(y <= -th)

        Dt_exc = np.diff(r)

        msDt[i,0] = np.mean(Dt_exc)
        msDt[i,1] = np.std(Dt_exc) / np.sqrt(len(r))
        msDt[i,2] = len(Dt_exc) / tot * 100
        msDt[i,3] = np.median(r) / (N/2) - 1
        msDt[i,4] = np.mean(r) / (N/2) -1
        msDt[i,5] = np.std(r) / np.sqrt(len(r))

        return msDt

#@numba.jit(nopython=True,parallel=True)
def DN_Burstiness(y):

    if y.mean() == 0:

        return np.nan

    r = np.std(y) / y.mean()

    B = ( r - 1 ) / ( r + 1 )

    return(B)

def DK_crinkle(x):

    x = x - np.mean(x)

    x2 = np.mean(np.square(x))**2

    if x2 == 0:
        return 0

    d2 = 2*x[1:-1] - x[0:-2] - x[2:]

    return np.mean(np.power(d2,4)) / x2

#@numba.jit(nopython=True,parallel=True)
#oddly this function slows down with numba
def DN_pleft(y,th = .1):

    p  = np.quantile(np.absolute(y - np.mean(y)),1-th)


    return p / np.std(y,ddof = 1)

def CO_FirstZero(y,corrFun = 'ac'):

    acf = CO_AutoCorr(y,[],'Fourier')

    N = len(y)

    for i in range(1,N-1):

        if acf[i] < 0:

            return i

    return N

def DN_Fit_mle(y,fitWhat = 'gaussian'):
    if fitWhat == 'gaussian':
        phat = stats.norm.fit(y)
        out = {'mean':phat[0],'std':phat[1]}
        return out
    else:
        print('Use gaussian geometric not implemented yet')

def CO_FirstMin(y, minWhat = 'ac'):

    if minWhat == 'mi':

        x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]

        acf = IN_AutoMutualInfo(y,x,'gaussian')

    else:

        acf = CO_AutoCorr(y,[],'Fourier')

    N = len(y)


    for i in range(1,N-1):

        if i == 2 and (acf[2] > acf[1]):

            return 1

        elif (i > 2) and (acf[i-2] > acf[i-1]) and (acf[i-1] < acf[i]):

            return i-1

    return N

def SY_DriftingMean(y,howl = 'num',l = ''):

    N = len(y)

    if howl == 'num':

        if l != '':

            l = math.floor(N/l)

    if l == '':

        if howl == 'num':

            l = 5

        elif howl == 'fix':

            l = 200

    if l == 0 or N < l:

        return

    numFits = math.floor(N / l)
    z = np.zeros((l,numFits))

    for i in range(0,numFits):

        z[:,i] = y[i*l :(i + 1)*l]



    zm = np.mean(z,axis = 0)
    zv = np.var(z,axis = 0,ddof = 1)

    meanvar = np.mean(zv)
    maxmean = np.max(zm)
    minmean = np.min(zm)
    meanmean = np.mean(zm)

    outDict = {}

    outDict['max'] = maxmean/meanvar
    outDict['min'] = minmean/meanvar
    outDict['mean'] = meanmean/meanvar
    outDict['meanmaxmin'] = (outDict['max']+outDict['min'])/2
    outDict['meanabsmaxmin'] = (np.absolute(outDict['max'])+np.absolute(outDict['min']))/2


    return outDict


import numpy as np
import scipy as sc
from scipy import stats
import math
import scipy.io # only needed if you uncomment testing code to compare with matlab (circumvents differences in random permutation between python and MATLAB)

# HELPER FILES REQUIRED
import Periphery



def FC_Suprise( y, whatPrior='dist', memory=0.2, numGroups=3, coarseGrainMethod='quantile', numIters=500, randomSeed='default'):
    '''
    How surprised you would be of the next data point given recent memory.

    Coarse-grains the time series, turning it into a sequence of symbols of a
    given alphabet size, numGroups, and quantifies measures of surprise of a
    process with local memory of the past memory values of the symbolic string.

    We then consider a memory length, memory, of the time series, and
    use the data in the proceeding memory samples to inform our expectations of
    the following sample.

    The 'information gained', log(1/p), at each sample using expectations
    calculated from the previous memory samples, is estimated.


    :param y: the input time series
    :param whatPrior: the type of information to store in memory
            (i) 'dist' : the values of the time series in the previous memory
            (ii) 'T1' : the one-point transition probabiltiites in the pervious memory samples
            (iii) 'T2' : the two point transition probabilties in the memory samples

    :param memory: the memory length (either number of samples, or a proportion of the time-series length, if between 0 and 1
    :param numGroups: the number of groups to coarse-grain the time series into
    :param coarseGrainMethod: the coarse-graining, or symbolization method
            (i) 'quantile' : an equiprobable alphabet by the value of each time series datapoint
            (ii) 'updown' : an equiprobable alphabet by the value of incremental changes in the time-series values
            (iii) 'embed2quadrants' : 4-letter alphabet of the quadrant each data point resides in a two-dimensional embedding space

    :param numIters: the number of interations to repeat the procedure for
    :param randomSeed: whether (and how) to reset the random seed, using BF_ResetSeed
    :return: a dictionary containing summaries of this series of information gains, including: minimum, maximum, mean, median, lower and upper quartiles, and standard deviation
    '''

    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    # Check inputs and set defaults -- most defaults were set in the function declaration above
    #------------------------------------------------------------------------------------------------------------------------------------------------------

    if (memory > 0) and (memory < 1): #specify memory as a proportion of the time series length

        memory = int(np.round(memory*len(y)))

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # COURSE GRAIN
    # requires SB_CoarseGrain.py helper function
    #------------------------------------------------------------------------------------------------------------------------------------------------------

    yth = SB_CoarseGrain(y, coarseGrainMethod, numGroups) # a coarse-grained time series using the numbers 1:numgroups

    N = int(len(yth))

    #select random samples to test
    BF_ResetSeed(randomSeed) #in matlab randomSeed defaults to an empty array [] and is then converted to 'default', here it defaults to 'default'

    rs = np.random.permutation(int(N-memory)) + memory # can't do beginning of time series, up to memory

    rs = np.sort(rs[0:min(numIters,(len(rs)-1))])

    rs = np.array([rs]) # made into two dimensional array to match matlab and work with testing code directly below


    # UNCOMMENT CODE BELOW TO COMPARE TO MATLAB USING rr data, make sure 'rs_var.mat' is in same folder as test file ( it's the resulting matlab rs value when using the UVA0001_rr.mat)
    # data = scipy.io.loadmat('rs_var.mat', squeeze_me = False)
    # rs = np.asarray(data['rs'])
    # print("rs MATLAB: ", rs)

    # -------------------------------------------------------------------------------------------------------------------
    # COMPUTE EMPIRICAL PROBABILITIES FROM TIME SERIES
    #-------------------------------------------------------------------------------------------------------------------

    store = np.zeros([numIters, 1])

    for i in range(0, rs.size): # rs.size
        if whatPrior == 'dist':
            # uses the distribution up to memory to inform the next point

            p = np.sum(yth[np.arange(rs[0, i]-memory-1, rs[0, i]-1)] == yth[rs[0, i]-1])/memory # had to be careful with indexing, arange() works like matlab's : operator
            store[i] = p


        elif whatPrior == 'T1':
            # uses one-point correlations in memory to inform the next point

            # estimate transition probabilites from data in memory
            # find where in memory this has been observbed before, and preceded it

            memoryData = yth[rs[0, i] - memory - 1:rs[0, i]-1] # every outputted value should be one less than in matlab

            # previous data observed in memory here
            inmem = np.nonzero(memoryData[0:memoryData.size - 1] == yth[rs[0, i]-2])
            inmem = inmem[0] # nonzero outputs a tuple of two arrays for some reason, the second one of all zeros


            if inmem.size == 0:
                p = 0
            else:
                p = np.mean(memoryData[inmem + 1] == yth[rs[0, i]-1])

            store[i] = p

        elif whatPrior == 'T2':

            # uses two point correlations in memory to inform the next point

            memoryData = yth[rs[0, i] - memory - 1:rs[0, i]-1] # every outputted value should be one less than in matlab

            inmem1 = np.nonzero(memoryData[1:memoryData.size - 1] == yth[rs[0, i]-2])
            inmem1 = inmem1[0]

            inmem2 = np.nonzero(memoryData[inmem1] == yth[rs[0, i]-3])
            inmem2 = inmem2[0]


            if inmem2.size == 0:
                p = 0
            else:
                p = np.sum(memoryData[inmem2+2] == yth[rs[0, i]-1])/len(inmem2)

            store[i] = p

        else:
            print("Error: unknown method: " + whatPrior)
            return

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # INFORMATION GAINED FROM NEXT OBSERVATION IS log(1/p) = -log(p)
    #-------------------------------------------------------------------------------------------------------------------------------------------

    store[store == 0] = 1 # so that we set log[0] == 0

    out = {} # dictionary for outputs

    for i in range(0, len(store)):
        if store[i] == 0:
            store[i] = 1

    store = -(np.log(store))

    #minimum amount of information you can gain in this way
    if np.any(store > 0):
        out['min'] = min(store[store > 0]) # find the minimum value in the array, excluding zero
    else:
        out['min'] = np.nan

    out['max'] = np.max(store) # maximum amount of information you cna gain in this way
    out['mean'] = np.mean(store)
    out['sum'] = np.sum(store)
    out['median'] = np.median(store)
    lq = sc.stats.mstats.mquantiles(store, 0.25, alphap=0.5, betap=0.5) # outputs an array of size one
    out['lq'] = lq[0] #convert array to int
    uq = sc.stats.mstats.mquantiles(store, 0.75, alphap=0.5, betap=0.5)
    out['uq'] = uq[0]
    out['std'] = np.std(store)

    if out['std'] == 0:
        out['tstat'] = np.nan
    else:
        out['tstat'] = abs((out['mean']-1)/(out['std']/math.sqrt(numIters)))

    return out # returns a dict with all of the output instead of a struct like in matlab, python doesnt have structs

def DK_theilerQ(x):
    x2 = np.mean(np.square(x))**(3/2)

    if x2 == 0:
        return 0

    d2 = x[0:-1] + x[1:]
    Q = np.mean(np.power(d2,3)) / x2

    return Q

def DN_IQR(y):
    return stats.iqr(y)

def EX_MovingThreshold(y,a = 1,b = .1):

    if b < 0 or b > 1:

        print("b must be between 0 and 1")
        return None

    N = len(y)
    y = np.absolute(y)
    q = np.zeros(N)
    kicks = np.zeros(N)

    q[0] = 1

    for i in range(1,N):

        if y[i] > q[i-1]:

            q[i] = (1 + a) * y[i]
            kicks[i] = q[i] - q[i-1]

        else:

            q[i] = ( 1 - b ) *  q[i-1]


    outDict = {}

    outDict['meanq'] = np.mean(q)
    outDict['medianq'] = np.median(q)
    outDict['iqrq'] = stats.iqr(q)
    outDict['maxq'] = np.max(q)
    outDict['minq'] = np.min(q)
    outDict['stdq'] = np.std(q)
    outDict['meanqover'] = np.mean(q - y)



    outDict['pkick'] = np.sum(kicks) / N - 1
    fkicks = np.argwhere(kicks > 0).flatten()
    Ikicks = np.diff(fkicks)
    outDict['stdkicks'] = np.std(Ikicks)
    outDict['meankickf'] = np.mean(Ikicks)
    outDict['mediankicksf'] = np.median(Ikicks)

    return outDict

def DN_CompareKSFit(x,whatDist = 'norm'):
    xStep = np.std(x) / 100
    if whatDist == 'norm':
        a, b = stats.norm.fit(x)
        peak = stats.norm.pdf(a,a,b)
        thresh = peak / 100
        xf1 = np.mean(x)
        ange = 10
        while ange > thresh:
            xf1 = xf1 - xStep
            ange = stats.norm.pdf(xf1,a,b)
        ange = 10
        xf2 = np.mean(x)
        while ange > thresh:
            xf2 = xf2 + xStep
            ange = stats.norm.pdf(xf2,a,b)


    #since some outliers real far away can take long time
    #should probably do pre-proccessing before functions
    if whatDist == "uni":

        a,b = stats.uniform.fit(x)
        peak = stats.uniform.pdf(np.mean(x),a,b-a)
        thresh = peak / 100
        xf1 = np.mean(x)
        ange = 10
        while ange > thresh:
            xf1 = xf1 - xStep
            ange = stats.norm.pdf(xf1,a,b)
        ange = 10
        xf2 = np.mean(x)
        while ange > thresh:
            xf2 = xf2 + xStep
            ange = stats.norm.pdf(xf2,a,b)

    #might over write y since changing x
    if whatDist == 'beta':
        scaledx = (x - np.min(x) + .01*np.std(x)) / (np.max(x)-np.min(x)+.02*np.std(x))
        xStep = np.std(scaledx) /100
        a = stats.beta.fit(scaledx)
        b = a[2]
        a = a[1]
        thresh = 1E-5
        xf1 = np.mean(scaledx)
        ange = 10
        while ange > thresh:
            xf1 = xf1 - xStep
            ange = stats.beta.pdf(xf1,a,b)
        ange = 10
        xf2 = np.mean(scaledx)
        while ange > thresh:
            xf2 = xf2 + xStep
            ange = stats.beta.pdf(xf2,a,b)
        x = scaledx


    kde = stats.gaussian_kde(x)
    test_space = np.linspace(np.min(x),np.max(x),1000)
    kde_est = kde(test_space)
    if whatDist == 'norm':
        ffit = stats.norm.pdf(test_space,a,b)
    if whatDist == 'uni':
        ffit = stats.uniform.pdf(test_space,a,b-a)
    if whatDist == 'beta':
        ffit = stats.beta.pdf(test_space,a,b)

    out = {}

    out['adiff'] = sum(np.absolute(kde_est - ffit)*(test_space[1]-test_space[0]))

    out['peaksepy'] = np.max(ffit) - np.max(kde_est)

    r = (ffit != 0)

    out['relent'] = sum(np.multiply(kde_est[r],np.log(np.divide(kde_est[r],ffit[r])))*(test_space[1]-test_space[0]))

    return out

from scipy import stats
def DN_Mode(y):
    #y must be numpy array
    if not isinstance(y,np.ndarray):
        y = np.asarray(y)
    return float(stats.mode(y).mode)

import numpy as np
import warnings
#import numba

#@numba.jit(nopython=True,parallel=True)
def EN_SampEn(x,m=2,r=.2,scale=True):
    warnings.filterwarnings('ignore')

    if scale:

        r = np.std(x,ddof = 1) * r

    print(r)

    templates = make_templates(x,m)
    #print(templates)
    A = 0
    B = 0

    for i in range(templates.shape[0]):

        template = templates[i,:]

        A = A + np.sum(np.amax(np.absolute(templates-template), axis=1) < r) -1

        B = B + np.sum(np.amax(np.absolute(templates[:,0:m]-template[0:m]),axis=1) < r) - 1

    if B == 0:

        return {'Sample Entropy':np.nan,"Quadratic Entropy":np.nan}


    return {'Sample Entropy':- np.log(A/B),"Quadratic Entropy": - np.log(A/B) + np.log(2*r)}

#@numba.jit(nopython=True,parallel=True)
def make_templates(x,m):

    N = int(len(x) - (m))

    templates = np.zeros((N,m+1))

    for i in range(N):

        templates[i,:] = x[i:i+m+1]

    return templates




# def SampEn(U, m = 2, r = .2):
#
#     r = r * np.log(U)
#
#     def _maxdist(x_i, x_j):
#
#         result = max([abs(ua - va) for ua, va in zip(x_i, x_j)])
#
#         return result
#
#
#     def _phi(m):
#
#         x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
#
#         C = 0
#
#         for i in range(len(x)):
#
#             for j in range(len(x)):
#
#                 if i == j:
#
#                     continue
#
#                 C += (_maxdist(x[i], x[j]) <= r)
#
#         return C
#
#
#     N = len(U)
#
#     return -np.log(_phi(m+1) / _phi(m))

# def EN_SampEn(y,M = 2,r = 0,pre = ''):
#     if r == 0:
#         r = .1*np.std(y)
#     else:
#         r = r*np.std(y)
#     M = M + 1
#     N = len(y)
#     print('hi')
#     lastrun = np.zeros(N)
#     run = np.zeros(N)
#     A = np.zeros(M)
#     B = np.zeros(M)
#     p = np.zeros(M)
#     e = np.zeros(M)
#
#     for i in range(1,N):
#         y1 = y[i-1]
#
#         for jj in range(1,N-i + 1):
#
#             j = i + jj - 1
#
#             if np.absolute(y[j] - y1) < r:
#
#                 run[jj] = lastrun[jj] + 1
#                 M1 = min(M,run[jj])
#                 for m in range(int(M1)):
#                     A[m] = A[m] + 1
#                     if j < N:
#                         B[m] = B[m] + 1
#             else:
#                 run[jj] = 0
#         for j in range(N-1):
#             lastrun[j] = run[j]
#
#     NN = N * (N - 1) / 2
#     p[0] = A[0] / NN
#     e[0] = - np.log(p[0])
#     for m in range(1,int(M)):
#         p[m] = A[m] / B[m-1]
#         e[m] = -np.log(p[m])
#     i = 0
#     # out = {'sampen':np.zeros(len(e)),'quadSampEn':np.zeros(len(e))}
#     # for ent in e:
#     #     quaden1 = ent + np.log(2*r)
#     #     out['sampen'][i] = ent
#     #     out['quadSampEn'][i] = quaden1
#     #     i = i + 1
#     out = {'Sample Entropy':e[1],'Quadratic Entropy':e[1] + np.log(2*r)}
#     return out

from scipy import signal
def SY_Trend(y):

    N  = len(y)
    stdRatio = np.std(signal.detrend(y)) / np.std(y)

    gradient, intercept = LinearFit(np.arange(N),y)

    yC = np.cumsum(y)
    meanYC = np.mean(yC)
    stdYC = np.std(yC)

    #print(gradient)
    #print(intercept)

    gradientYC, interceptYC = LinearFit(np.arange(N),yC)

    meanYC12 = np.mean(yC[0:int(np.floor(N/2))])
    meanYC22 = np.mean(yC[int(np.floor(N/2)):])

    out = {'stdRatio':stdRatio,'gradient':gradient,'intercept':intercept,
            'meanYC':meanYC,'stdYC':stdYC,'gradientYC':gradientYC,
            'interceptYC':interceptYC,'meanYC12':meanYC12,'meanYC22':meanYC22}
    return out

def LinearFit(xData,yData):
    m, b = np.polyfit(xData,yData,1)
    return m,b

import numpy as np


#@numba.jit(nopython=True,parallel=True)
def DN_Mean(y):
    #y must be numpy array
    # if not isinstance(y,np.ndarray):
    #     y = np.asarray(y)
    return(y.mean())

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

def DN_Cumulants(y,cumWhatMay = 'skew1'):
    if cumWhatMay == 'skew1':
        return stats.skew(y)
    elif cumWhatMay == 'skew2':
        return stats.skew(y,0)
    elif cumWhatMay == 'kurt1':
        return stats.kurtosis(y)
    elif cumWhatMay == 'kurt2':
        return stats.kurtosis(y,0)
    else:
         raise Exception('Requested Unknown cumulant must be: skew1, skew2, kurt1, or kurt2')

def DN_Range(y):
    return np.max(y) - np.min(y)

from Periphery import *
def DN_FitKernalSmooth(x,varargin = {}):
    #varargin should be dict with possible keys numcross
    #area and arclength

    out = {}

    m = np.mean(x)

    kde = stats.gaussian_kde(x)
    #i think matlabs kde uses 100 points
    #but end numbers end up being midly off
    #seems to be rounding entropy max, min line up
    test_space = np.linspace(np.min(x),np.max(x),100)

    f = kde(test_space)

    df = np.diff(f)

    ddf  = np.diff(df)

    sdsp = ddf[BF_sgnchange(df,1)]

    out['npeaks'] = sum(sdsp < -.0002)

    out['max'] = np.max(f)

    out['entropy'] = - sum(np.multiply(f[f>0],np.log(f[f>0])))*(test_space[2]-test_space[1])

    out1 = sum(f[test_space > m]) * (test_space[2]-test_space[1])
    out2 = sum(f[test_space < m]) * (test_space[2]-test_space[1])
    out['asym'] = out1 / out2

    out1 = sum(np.absolute(np.diff(f[test_space < m]))) * (test_space[2]-test_space[1])
    out1 = sum(np.absolute(np.diff(f[test_space > m]))) * (test_space[2]-test_space[1])
    out['plsym'] = out1 / out2

    if 'numcross' in varargin:
        thresholds = varargin['numcross']
        out['numCrosses']  = {}
        for i in range(len(thresholds)):
            numCrosses = sum(BF_sgnchange(f - thresholds[i]))
            out['numCrosses'][thresholds[i]] = numCrosses
    if 'area' in varargin:
        thresholds = varargin['area']
        out['area']  = {}
        for i in range(len(thresholds)):
            areaHere = sum(f[f < thresholds[i]]) * (test_space[2]-test_space[1])
            out['area'][thresholds[i]] = areaHere
    if 'arclength' in varargin:
        thresholds = varargin['arclength']
        out['arclength']  = {}
        for i in range(len(thresholds)):
            fd = np.absolute(np.diff(f[(test_space > m - thresholds[i]) & (test_space < m + thresholds[i])]))
            arclengthHere = sum(fd) * (test_space[2]-test_space[1])
            out['arclength'][thresholds[i]] = arclengthHere
    return out

import numpy as np
#@numba.jit(nopython=True)
def DN_Median(y):
    #y must be numpy array
    # if not isinstance(y,np.ndarray):
    #     y = np.asarray(y)
    return(np.median(y))


import numpy as np

def SY_StdNthDer(y, n=2):
    '''
    SY_StdNthDer  Standard deviation of the nth derivative of the time series.

    Based on an idea by Vladimir Vassilevsky, a DSP and Mixed Signal Design
    Consultant in a Matlab forum, who stated that You can measure the standard
    deviation of the nth derivative, if you like".

    cf. http://www.mathworks.de/matlabcentral/newsreader/view_thread/136539

    The derivative is estimated very simply by simply taking successive increments
    of the time series; the process is repeated to obtain higher order
    derivatives.

    Note that this idea is popular in the heart-rate variability literature, cf.
    cf. "Do Existing Measures ... ", Brennan et. al. (2001), IEEE Trans Biomed Eng 48(11)
    (and function MD_hrv_classic)

    :param y: time series to analyze
    :param n: the order of derivative to analyze
    :return: the standard deviation of the nth derivative of the time series
    '''

    yd = np.diff(y, n) # approximate way to calculate derivative

    if yd.size is 0:
        print("Time series too short to compute differences")

    out = np.std(yd, ddof=1)

    return out

def SY_DynWin(y,maxnseg = 10):

    nsegr = np.arange(2,maxnseg + 1)

    nmov = 1

    numFeatures = 9

    fs = np.zeros((len(nsegr),numFeatures))

    taug = CO_FirstZero(y,'ac')

    for i in range(len(nsegr)):

        nseg = nsegr[i]

        wlen = math.floor( len(y) / nseg )

        inc = math.floor( wlen / nmov )

        if inc == 0:

            inc = 1

        numSteps = math.floor((len(y) - wlen) / inc) + 1

        qs = np.zeros((numSteps,numFeatures))

        for j in range(numSteps):

            ysub = y[(j)*inc:(j)*inc + wlen]

            taul = CO_FirstZero(ysub,'ac')

            qs[j,0] = np.mean(ysub)

            qs[j,1] = np.std(ysub,ddof = 1)

            qs[j,2] = stats.skew(ysub)

            qs[j,3] = stats.kurtosis(ysub)

            qs[j,4] = CO_AutoCorr(ysub,1,'Fourier')

            qs[j,5] = CO_AutoCorr(ysub,2,'Fourier')

            qs[j,6] = CO_AutoCorr(ysub,taug,'Fourier')

            qs[j,7] = CO_AutoCorr(ysub,taul,'Fourier')

            qs[j,8]  = taul

        fs[i,:] = np.std(qs,axis = 0,ddof = 1)



    fs = np.std(fs,axis  = 0,ddof = 1)


    outDict = {}

    outDict['stdmean'] = fs[0]
    outDict['stdstd'] = fs[1]
    outDict['stdskew'] = fs[2]
    outDict['stdkurt'] = fs[3]
    outDict['stdac1'] = fs[4]
    outDict['stdac2'] = fs[5]
    outDict['stdactaug'] = fs[6]
    outDict['stdactaul'] = fs[7]
    outDict['stdtaul'] = fs[8]


    return outDict

#@numba.jit(nopython=True,parallel=True)
def DN_Spread(y,spreadMeasure = 'std'):
    if spreadMeasure == 'std':
        return np.std(y)
    elif spreadMeasure == 'iqr':
        return stats.iqr(y)
    elif spreadMeasure == 'mad':
        return mad(y)
    elif spreadMeasure == 'mead':
        return mead(y)#stats.median_absolute_deviation(y)
    else:
        raise Exception('spreadMeasure must be one of std, iqr, mad or mead')
def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

def mead(data, axis=None):
    return np.median(np.absolute(data - np.median(data, axis)), axis)

#@numba.jit(nopython=True,parallel=True)
def DN_MinMax(y,which = 'max'):
    # if not isinstance(y,np.ndarray):
    #     y = np.asarray(y)
    if which == 'min':
        return(y.min())
    else:
        return(y.max())

def DN_OutlierTest(y,p = 2,justMe=''):

    outDict = {}

    index = np.logical_and(y > np.percentile(y,p),y < np.percentile(y,100-p))

    outDict['mean'] = np.mean(y[index])

    outDict['std'] = np.std(y[index],ddof = 1) / np.std(y,ddof = 1)

    if justMe == 'mean':

        return outDict['mean']

    elif justMe == 'std':

        return outDict['std']

    return outDict

def MD_polvar(x,d = 1 ,D = 6):

    dx = np.absolute(np.diff(x))

    N = len(dx)

    xsym = ( dx >= d )
    zseq = np.zeros(D)
    oseq = np.ones(D)

    i = 1
    pc = 0

    while i <= (N-D):

        xseq = xsym[i:(i+D)]

        if np.sum((xseq == zseq)) == D or np.sum((xseq == oseq)) == D:

            pc = pc + 1
            i = i + D

        else:

            i = i + 1

    p = pc / N

    return p

def DK_timerev(x,timeLag = 1):

    foo = DK_lagembed(x,3,timeLag)

    a = foo[:,0]
    b = foo[:,1]
    c = foo[:,2]

    res = np.mean(np.multiply(np.multiply(a,a),b) - np.multiply(np.multiply(b,c),c))

    return res

def CO_RM_AMInformation(y,tau = 1):
    """
    A wrapper for rm_information(), which calculates automutal information

    Inputs:
        y, the input time series
        tau, the time lag at which to calculate automutal information

    :returns estimate of mutual information

    - Wrapper initially developed by Ben D. Fulcher in MATLAB
    - rm_information.py initially developed by Rudy Moddemeijer in MATLAB
    - Translated to python by Tucker Cullen

    """

    if tau >= len(y):

        return

    y1 = y[:-tau]
    y2 = y[tau:]

    out = RM_information(y1,y2)

    return out[0]

#@numba.jit(nopython=True,parallel=True)
def DN_CustomSkewness(y,whatSkew = 'pearson'):
    if whatSkew == 'pearson':
        if np.std(y) != 0:
            return (3*np.mean(y) - np.median(y)) / np.std(y)
        else:
            return 0
    elif whatSkew == 'bowley':
        qs = np.quantile(y,[.25,.5,.75])
        if np.std(y) != 0:
            return (qs[2] + qs[0] - 2*qs[1]) / (qs[2] - qs[0])
        else:
            return 0

    else:
         raise Exception('whatSkew must be either pearson or bowley.')

def EN_mse(y,scale=range(2,11),m=2,r=.15,adjust_r=True):

    minTSLength = 20
    numscales = len(scale)
    y_cg = []

    for i in range(numscales):

        bufferSize = scale[i]
        y_buffer = BF_makeBuffer(y,bufferSize)
        y_cg.append(np.mean(y_buffer,1))

    outEns = []



    for si in range(numscales):
        if len(y_cg[si]) >= minTSLength:

            sampEnStruct = EN_SampEn(y_cg[si],m,r)
            outEns.append(sampEnStruct)
        else:
            outEns.append(np.nan)
    sampEns = []
    for out in outEns:
        if not isinstance(out,dict):
            sampEns.append(np.nan)
            continue
        sampEns.append(out['Sample Entropy'])

    maxSampen = np.max(sampEns)
    maxIndx = np.argmax(sampEns)

    minSampen = np.min(sampEns)
    minIndx = np.argmin(sampEns)

    meanSampen = np.mean(sampEns)

    stdSampen = np.std(sampEns)

    meanchSampen = np.mean(np.diff(sampEns))

    out = {'max Samp En':maxSampen,'max point':scale[maxIndx],'min Samp En':minSampen,\
    'min point':scale[minIndx],'mean Samp En':meanSampen,'std Samp En':stdSampen, 'Mean Change':meanchSampen}

    i = 1
    for sampEn in sampEns:
        out['sampEn ' + str(i)] = sampEn
        i = i + 1

    return out

def IN_AutoMutualInfo(y,timeDelay = 1,estMethod = 'gaussian',extraParam = []):

    if isinstance(timeDelay,str):

        timeDelay = CO_FirstZero(y,'ac')

    N = len(y)

    if isinstance(timeDelay,list):

        numTimeDelays = len(timeDelay)

    else:

        numTimeDelays = 1

        timeDelay = [timeDelay]

    amis = []

    out = {}

    for k in range(numTimeDelays):

        y1 = y[0:N-timeDelay[k]]
        y2 = y[timeDelay[k]:N]

        if estMethod == 'gaussian':

            r = np.corrcoef(y1,y2)[1,0]

            amis.append(-.5 * np.log(1 - r**2))

            out['Auto Mutual ' + str(timeDelay[k])] = -.5 * np.log(1 - r**2)

    return out

def SB_MotifTwo(y,binarizeHow = 'diff'):

    yBin = BF_Binarize(y,binarizeHow)

    N = len(yBin)

    r1 = (yBin == 1)

    r0 = (yBin == 0)

    outDict = {}

    outDict['u'] = np.mean(r1)
    outDict['d'] = np.mean(r0)

    pp  = np.asarray([ np.mean(r1), np.mean(r0)])

    outDict['h'] = f_entropy(pp)

    r1 = r1[:-1]
    r0 = r0[:-1]

    rs1 = [r0,r1]

    rs2 = [[0,0],[0,0]]
    pp = np.zeros((2,2))

    letters = ['d','u']

    for i in range(2):

        l1 = letters[i]

        for j in range(2):

            l2 = letters[j]

            rs2[i][j] = np.logical_and(rs1[i],yBin[1:] == j)

            outDict[l1 + l2] = np.mean(rs2[i][j])

            pp[i,j] = np.mean(rs2[i][j])

            rs2[i][j] = rs2[i][j][:-1]

    outDict['hh'] = f_entropy(pp)

    rs3 = [[[0,0],[0,0]],[[0,0],[0,0]]]
    pp = np.zeros((2,2,2))

    for i in range(2):

        l1 = letters[i]

        for j in range(2):

            l2 = letters[j]

            for k in range(2):

                l3 = letters[k]

                rs3[i][j][k] =np.logical_and(rs2[i][j],yBin[2:] == k)

                outDict[l1 + l2 + l3] = np.mean(rs3[i][j][k])

                pp[i,j,k] = np.mean(rs3[i][j][k])

                rs3[i][j][k] = rs3[i][j][k][:-1]

    outDict['hhh'] = f_entropy(pp)

    rs4 = [[[[0,0],[0,0]],[[0,0],[0,0]]],[[[0,0],[0,0]],[[0,0],[0,0]]]]
    pp = np.zeros((2,2,2,2))

    for i in range(2):

        l1 = letters[i]

        for j in range(2):

            l2 = letters[j]

            for k in range(2):

                l3 = letters[k]

                for l in range(2):

                    l4 = letters[l]

                    rs4[i][j][k][l] =np.logical_and(rs3[i][j][k],yBin[3:] == l)

                    outDict[l1 + l2 + l3 + l4] = np.mean(rs4[i][j][k][l])

                    pp[i,j,k,l] = np.mean(rs4[i][j][k][l])


    outDict['hhhh'] = f_entropy(pp)


    return outDict

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

def EN_CID(y):


    CE1 = f_CE1(y)
    CE2 = f_CE2(y)

    minCE1 = f_CE1(np.sort(y))
    minCE2 = f_CE2(np.sort(y))

    CE1_norm = CE1 / minCE1
    CE2_norm = CE2 / minCE2

    out = {'CE1':CE1,'CE2':CE2,'minCE1':minCE1,'minCE2':minCE2,
            'CE1_norm':CE1_norm,'CE2_norm':CE2_norm}
    return out

def f_CE1(y):
    return np.sqrt(np.mean(np.power(np.diff(y),2) ) )

def f_CE2(y):
    return np.mean(np.sqrt( 1 + np.power(np.diff(y),2) ) )

def PH_ForcePotential(y,whatPotential = 'dblwell',params = []):

    if params == []:

        if whatPotential == 'dblwell':

            params = [2,.1,.1]

        elif whatPotential == 'sine':

            params = [1,1,1]

        else:

            print('Unreconginzed Potential')
            return

    N  = len(y)

    if len(params) < 3:

        print('3 Parameters required')
        return

    alpha = params[0]
    kappa = params[1]
    deltat = params[2]

    if whatPotential == 'dblwell':

        def F(x): return - np.power(x,3) + alpha**2 * x

        def V(x): return np.power(x,4) / 4 - alpha**2 * np.power(x,2) / 2

    if whatPotential == 'sine':

        def F(x): return np.sin(x / alpha) / alpha

        def V(x): return - np.cos(x / alpha)

    x = np.zeros(N)
    v = np.zeros(N)

    for i in range(1,N):

        x[i] =  x[i-1] + v[i-1]*deltat+(F(x[i-1])+y[i-1]-kappa*v[i-1])*deltat**2
        v[i] = v[i-1] + (F(x[i-1])+y[i-1]-kappa*v[i-1])*deltat

    if np.isnan(x[-1]) or np.absolute(x[-1]) > 1000000000:

        print('Trajectroy Blew out!')
        return

    outDict = {}

    outDict['mean'] = np.mean(x)
    outDict['median'] = np.median(x)
    outDict['std'] = np.std(x)
    outDict['range'] = np.max(x) - np.min(x)
    outDict['proppos'] = np.sum((x > 0)) / N
    outDict['pcross'] = np.sum((np.multiply(x[:-1],x[1:]) < 0)) / ( N - 1 )
    outDict['ac1'] = np.absolute(CO_AutoCorr(x,1,'Fourier'))
    outDict['ac10'] = np.absolute(CO_AutoCorr(x,10,'Fourier'))
    outDict['ac25'] = np.absolute(CO_AutoCorr(x,25,'Fourier'))
    outDict['tau'] = CO_FirstZero(x,'ac')
    outDict['finaldev'] = np.absolute(x[-1])

    if whatPotential == 'dblwell':

        outDict['pcrossup'] = np.sum((np.multiply(x[:-1] - alpha,x[1:]-alpha) < 0)) / (N-1)
        outDict['pcrossdown'] = np.sum((np.multiply(x[:-1] + alpha,x[1:]+alpha) < 0)) / (N-1)


    return outDict

def DN_Unique(x):
    return len(np.unique(x)) / len(x)

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

from scipy import optimize
def DT_IsSeasonal(y):

    N = len(y)

    th_fit = 0.3
    th_ampl = 0.5

    try:
        params, params_covariance = optimize.curve_fit(test_func, np.arange(N), y, p0=[10, 13,600,0])
    except:
        return False

    a,b,c,d = params



    y_pred = a * np.sin(b * np.arange(N) + d) + c

    SST = sum(np.power(y - np.mean(y),2))
    SSr = sum(np.power(y - y_pred,2))

    R = 1 - SSr / SST


    if R > th_fit: #and (np.absolute(a) > th_ampl*.1*np.std(y)):
        return True
    else:
        return False

def test_func(x, a, b,c,d):
    return a * np.sin(b * x + d) + c

def EN_DistributionEntropy(y,histOrKS = 'hist',numBins = None,olremp = 0):
    """
     EN_DistributionEntropy    Distributional entropy.

     Estimates of entropy from the distribution of a data vector. The
     distribution is estimated either using a histogram with numBins bins, or as a
     kernel-smoothed distribution, using the ksdensity function from Matlab's
     Statistics Toolbox with width parameter, w (specified as the iunput numBins).

     An optional additional parameter can be used to remove a proportion of the
     most extreme positive and negative deviations from the mean as an initial
     pre-processing.

    ---INPUTS:

     y, the input time series

     histOrKS: 'hist' for histogram, or 'ks' for ksdensity

     numBins: (*) (for 'hist'): an integer, uses a histogram with that many bins
              (*) (for 'ks'): a positive real number, for the width parameter for
                           ksdensity (can also be empty for default width
                                           parameter, optimum for Gaussian)

     olremp [opt]: the proportion of outliers at both extremes to remove
                   (e.g., if olremp = 0.01; keeps only the middle 98 of data; 0
                   keeps all data. This parameter ought to be less than 0.5, which
                   keeps none of the data).
                   If olremp is specified, returns the difference in entropy from
                   removing the outliers.

    Warning:
        Will not match matlab version exactly. Histogram binning is slightly
        different. Matlab uses edge numpy version uses center of bin

    """
    if not olremp == 0:

        index = np.logical_and(y >= np.quantile(y,olremp),y <= np.quantile(y,1-olremp))


        yHat = y[index]

        if len(yHat) == 0:

            return None

        else:

            return EN_DistributionEntropy(y,histOrKS,numBins) - \
                    EN_DistributionEntropy(yHat,histOrKS,numBins)

    if histOrKS == 'hist':

        if numBins is None:

            numBins = 10

        if isinstance(numBins,int):

            px, binEdges = np.histogram(y,numBins)

            px = px / sum(px)

        else:

            px, binEdges = np.histogram(y,numBins)

            px = px / sum(px)

        binWidths = np.diff(binEdges)

    if histOrKS == 'ks':
        #ks doesnt work for now kde python vs matlab different
        return None

        if numBins is None:

            kde = stats.gaussian_kde(y)

            print(kde(y))

        else:

            px, xr = stats.gaussian_kde(x,numBins)

    px2 = px[px > 0]

    binWidths2 = binWidths[px > 0]

    out = - np.sum( np.multiply(px2,np.log(np.divide(px2,binWidths2))) )

    return out

#@numba.jit(nopython=True,parallel=True)
def EN_ApEn(y,mnom = 1,rth = .2):

    r = rth * np.std(y)
    N = len(y)
    phi = np.zeros(2)

    for k in range(2):
        m = mnom + k
        m = int(m)
        C = np.zeros(N-m+1)

        x = np.zeros((N - m + 1, m))

        for i in range(N - m + 1):
            x[i,:] = y[i:i+m]

        ax = np.ones((N - m + 1, m))
        for i in range(N-m+1):

            for j in range(m):
                ax[:,j] = x[i,j]

            d = np.absolute(x-ax)
            if m > 1:
                d = np.maximum(d[:,0],d[:,1])
            dr = ( d <= r )
            C[i] = np.sum(dr) / (N-m+1)
        phi[k] = np.mean(np.log(C))
    return phi[0] - phi[1]

#import matplotlib.pyplot as plt
def SC_HurstExp(x):

    N = len(x)

    splits = int(np.log2(N))

    rescaledRanges = []

    n = []

    for i in range(splits):

        chunks = 2**(i)

        n.append(int(N / chunks))


        y = x[:N - N % chunks]

        y = y.reshape((chunks,int(N/chunks)))

        m = y.mean(axis = 1,keepdims = True)

        y = y - m

        z = np.cumsum(y,1)

        R = np.max(z,1) - np.min(z,1)

        S = np.std(y,1)

        S[S == 0] = 1


        rescaledRanges.append(np.mean(R/S))

    logRS = np.log(rescaledRanges)
    logn = np.log(n)

    # plt.plot(logn,logRS)
    # plt.show()

    p = np.polyfit(logn,logRS,1)

    return p[0]

def DN_ObsCount(y):
    return np.count_nonzero(~np.isnan(y))

#@numba.jit(nopython=True,parallel=True)
def EN_ShannonEn(y):
    p = np.zeros(len(np.unique(y)))
    n = 0
    for i in np.unique(y):
        p[n] = len(y[y == i]) / len(y)
        n = n + 1

    return -np.sum(p*np.log2(p))

# author: Dominik Krzeminski (dokato)
# https://github.com/dokato/dfa
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss

# detrended fluctuation analysis

def calc_rms(x, scale):
    """
    windowed Root Mean Square (RMS) with linear detrending.

    Args:
    -----
      *x* : numpy.array
        one dimensional data vector
      *scale* : int
        length of the window in which RMS will be calculaed
    Returns:
    --------
      *rms* : numpy.array
        RMS data in each window with length len(x)//scale
    """
    # making an array with data divided in windows
    shape = (x.shape[0]//scale, scale)
    X = np.lib.stride_tricks.as_strided(x,shape=shape)
    # vector of x-axis points to regression
    scale_ax = np.arange(scale)
    rms = np.zeros(X.shape[0])
    for e, xcut in enumerate(X):
        coeff = np.polyfit(scale_ax, xcut, 1)
        xfit = np.polyval(coeff, scale_ax)
        # detrending and computing RMS of each window
        rms[e] = np.sqrt(np.mean((xcut-xfit)**2))
    return rms

def dfa(x, scale_lim=[5,9], scale_dens=0.25, show=False):
    """
    Detrended Fluctuation Analysis - measures power law scaling coefficient
    of the given signal *x*.

    More details about the algorithm you can find e.g. here:
    Hardstone, R. et al. Detrended fluctuation analysis: A scale-free
    view on neuronal oscillations, (2012).

    Args:
    -----
      *x* : numpy.array
        one dimensional data vector
      *scale_lim* = [5,9] : list of length 2
        boundaries of the scale, where scale means windows among which RMS
        is calculated. Numbers from list are exponents of 2 to the power
        of X, eg. [5,9] is in fact [2**5, 2**9].
        You can think of it that if your signal is sampled with F_s = 128 Hz,
        then the lowest considered scale would be 2**5/128 = 32/128 = 0.25,
        so 250 ms.
      *scale_dens* = 0.25 : float
        density of scale divisions, eg. for 0.25 we get 2**[5, 5.25, 5.5, ... ]
      *show* = False
        if True it shows matplotlib log-log plot.
    Returns:
    --------
      *scales* : numpy.array
        vector of scales (x axis)
      *fluct* : numpy.array
        fluctuation function values (y axis)
      *alpha* : float
        estimation of DFA exponent
    """
    # cumulative sum of data with substracted offset
    y = np.cumsum(x - np.mean(x))
    scales = (2**np.arange(scale_lim[0], scale_lim[1], scale_dens)).astype(np.int)
    fluct = np.zeros(len(scales))
    # computing RMS for each window
    for e, sc in enumerate(scales):
        if len(calc_rms(y, sc)**2) == 0:
            continue
        fluct[e] = np.sqrt(np.mean(calc_rms(y, sc)**2))

    # fitting a line to rms data
    coeff = np.polyfit(np.log2(scales), np.log2(fluct), 1)
    # if show:
    #     fluctfit = 2**np.polyval(coeff,np.log2(scales))
    #     plt.loglog(scales, fluct, 'bo')
    #     plt.loglog(scales, fluctfit, 'r', label=r'$\alpha$ = %0.2f'%coeff[0])
    #     plt.title('DFA')
    #     plt.xlabel(r'$\log_{10}$(time window)')
    #     plt.ylabel(r'$\log_{10}$<F(t)>')
    #     plt.legend()
    #     plt.show()
    #return scales, fluct, coeff[0]
    return coeff[0]

from statsmodels.tsa.arima_model import ARIMA
def MF_AR_arcov(y,p = 2):


    try:

        model = ARIMA(y, order=(p,0,0))
        model_fit = model.fit( disp=False)

    except:
        #Non-stationary returns expception
        return

    ar_coefs = model_fit.arparams
    coef_errors = model_fit.bse

    outDict = {}

    stable = True

    for num in model_fit.arroots:

        if np.absolute(num) < 1:

            stable = False

    outDict['stable'] = stable

    y_est = model_fit.fittedvalues[-1]

    err = y - y_est

    for i in range(len(ar_coefs)):

        outDict['ar' + str(i  + 1)] = ar_coefs[i]
        outDict['ar error' + str(i + 1)] = coef_errors[i]

    outDict['res_mu'] = np.mean(err)
    outDict['res_std'] = np.std(err)

    outDict['res_AC1'] = CO_AutoCorr(err,1)
    outDict['res_AC2'] = CO_AutoCorr(err,2)

    return outDict

def CO_tc3(y,tau = 'ac'):

    if tau == 'ac':

        tau = CO_FirstZero(y,'ac')


    # else:
    #
    #     tau = CO_FirstMin(y,'mi')

    N = len(y)

    yn = y[0:N-2*tau]
    yn1 = y[tau:N-tau]
    yn2 = y[tau*2:N]

    try:
        raw = np.mean(np.multiply(np.multiply(yn,yn1),yn2)) / (np.absolute(np.mean(np.multiply(yn,yn1))) ** (3/2))
    except:
        return {'raw':np.nan,'abs':np.nan,'num':np.nan,'absnum':np.nan,'denom':np.nan}

    outDict = {}

    outDict['raw'] = raw

    outDict['abs'] = np.absolute(raw)

    outDict['num'] = np.mean(np.multiply(yn,np.multiply(yn1,yn2)))

    outDict['absnum'] = np.absolute(outDict['num'])

    outDict['denom'] = np.absolute( np.mean(np.multiply(yn,yn1)))**(3/2)

    return outDict

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

def SY_RangeEvolve(y):

    N = len(y)

    cums = np.zeros(N)

    for i in range(1,N+1):

        cums[i-1] = np.max(y[:i]) - np.min(y[:i])

    fullr = cums[N-1]

    outDict = {}

    outDict['totnuq'] = lunique(cums)

    outDict['nuqp1'] = cumtox(cums,.01,N,outDict['totnuq'])
    outDict['nuqp10'] = cumtox(cums,.1,N,outDict['totnuq'])
    outDict['nuqp20'] = cumtox(cums,.2,N,outDict['totnuq'])
    outDict['nuqp50'] = cumtox(cums,.5,N,outDict['totnuq'])

    Ns = [10,50,100,1000]

    for n in Ns:

        if N <= n:

            outDict['nuql' + str(n)] = np.nan


        else:

            outDict['nuql' + str(n)] = lunique(cums[:n]) / outDict['totnuq']


    Ns = [10,50,100,1000]

    for n in Ns:

        if N >= n:

            outDict['l' + str(n)] = cums[n - 1] / fullr


        else:

            outDict['l' + str(n)]= np.nan



    outDict['p1'] = cums[math.ceil(N*.01)-1] / fullr
    outDict['p10'] = cums[math.ceil(N*.1)-1] / fullr
    outDict['p20'] = cums[math.ceil(N*.2)-1] / fullr
    outDict['p50'] = cums[math.ceil(N*.5)-1] / fullr



    return outDict

def lunique(x):
    return len(np.unique(x))
def cumtox(cums,x,N,tot):
    return lunique(cums[:math.floor(N*x)]) / tot

def DN_nlogL_norm(y):
    muhat, sigmahat = stats.norm.fit(y)
    z = (y - muhat) / sigmahat
    L = -.5*np.power(z,2) - np.log(np.sqrt(2*math.pi)*sigmahat)
    return -sum(L) / len(y)


def CO_AutoCorr(y,lag = 1,method = 'Fourier',t=1):

    # if not isinstance(y,np.ndarray):
    #     y = np.asarray(y)


    if method == 'TimeDomainStat':

        if lag == []:

            acf = [1]

            for i in range(1,len(y)-1):

                acf.append(np.corrcoef(y[:-i],y[i:])[0,1])

            return acf

        return(np.corrcoef(y[:-lag],y[lag:])[0,1])

    else:

        N = len(y)



        nFFT = int(2**(np.ceil(np.log2(N)) + 1))

        F = np.fft.fft(y - y.mean(),nFFT)

        F = np.multiply(F,np.conj(F))

        acf = np.fft.ifft(F)

        if acf[0] == 0:

            if lag == []:

                return acf
            if lag > N:

                #print("Lag larger than series")

                return np.nan

            return acf[lag]

        acf = acf / acf[0]
        acf = acf.real

        if lag == []:

            return acf

        if lag > N:

            #print("Lag larger than series")

            return np.nan
        try:
            return acf[lag]
        except:
            print('y is: \n\n\n\n\n\n\n\n\n\n\n')
            print(y)
            return acf[lag]

import arch
from pprint import pprint

from statsmodels.stats.diagnostic import het_arch

def MF_GARCHFit(y,preproc = None,P = 1,Q = 1):

    y = (y - np.mean(y)) / np.std(y)

    N = len(y)

    outDict = {}

    lm, lmpval,fval,fpval = het_arch(y)

    outDict['lm'] = lm
    outDict['lmpval'] = lmpval
    outDict['fval'] = fval
    outDict['fpval'] = fpval

    model= arch.arch_model(y, vol='Garch', p=P, o=0, q=Q, dist='Normal')
    results=model.fit()

    #print(results.summary())

    params = results._params
    paraNames = results._names

    outDict['logl'] = results._loglikelihood
    outDict['success'] = results._optim_output['success']

    for i in range(len(params)):

        outDict[paraNames[i]] = params[i]

    #pprint(vars(results))

    return outDict

import math
def CO_f1ecac(y):

    N = len(y)

    thresh = 1 / math.exp(1)

    for i in range(1,N):

        auto = CO_AutoCorr(y,i)

        if ( auto - thresh ) < 0:

            return i

    return N

def EN_wentropy(y,whaten = 'shannon',p = ''):

    N = len(y)

    if whaten == 'shannon':

        out = wentropy(y,'shannon') / N

    elif whaten == 'logenergy':

        out = wentropy(y,'logenergy') / N

    elif whaten == 'threshold':

        if p == '':

            p = np.mean(y)

        out = wentropy(y,'threshold',p) / N

    elif whaten == 'sure':

        if p == '':

            p = np.mean(y)

        out = wentropy(y,'sure',p) / N

    elif whaten == 'norm':
        if p == '':

            p = 2

        out = wentropy(y,'norm',p) / N

    else:

        out = None

    return out

def DN_ProportionValues(x,propWhat = 'positive'):
    N = len(x)
    if propWhat == 'zeros':
        return sum(x == 0) / N
    elif propWhat == 'positive':
        return sum(x > 0) / N
    elif propWhat == 'negative':
        return sum(x < 0) / N
    else:
        raise Exception('Only negative, positve, zeros accepted for propWhat.')

def NL_BoxCorrDim(y,numBins = 100,embedParams = ['ac','fnnmar']):
    pass
    # if len embedParams <

from matplotlib import mlab
def SY_PeriodVital(x):

    f1 = 1
    f2 = 6

    z = np.diff(x)

    [F, t, p] =  signal.spectrogram(z,fs = 60)

    f = np.logical_and(F >= f1,F <= f2)

    p = p[f]

    F = F[f]

    Pmean = np.mean(p)

    Pmax = np.max(p)
    ff = np.argmax(p)
    if ff >= len(F):
        Pf = np.nan
    else:
        Pf = F[ff]
    Pr = Pmax / Pmean
    Pstat = np.log(Pr)

    return {'Pstat':Pstat,'Pmax':Pmax,'Pmean':Pmean,'Pf':Pf}


import numpy as np
import scipy
from scipy import signal
import math

def MD_hrv_classic(y):
    """

    classic heart rate variabilty statistics

    Typically assumes an NN/RR time series in the units of seconds

    :param y: the input time series

    Includes:
    (i) pNNx
    cf. "The pNNx files: re-examining a widely used heart rate variability
           measure", J.E. Mietus et al., Heart 88(4) 378 (2002)

    (ii) Power spectral density ratios in different frequency ranges
    cf. "Heart rate variability: Standards of measurement, physiological
        interpretation, and clinical use",
        M. Malik et al., Eur. Heart J. 17(3) 354 (1996)

    (iii) Triangular histogram index, and

    (iv) Poincare plot measures
    cf. "Do existing measures of Poincare plot geometry reflect nonlinear
       features of heart rate variability?"
        M. Brennan, et al., IEEE T. Bio.-Med. Eng. 48(11) 1342 (2001)

    Code is heavily derived from that provided by Max A. Little:
    http://www.maxlittle.net/

    """

    #Standard Defaults
    diffy = np.diff(y)
    N = len(y)

    # Calculate pNNx percentage ---------------------------------------------------------------------------------
    Dy = np.abs(diffy) * 1000

    # anonymous function to fo the PNNx calculation:
    # proportion of the difference magnitudes greater than X*sigma
    PNNxfn = lambda x : np.sum(Dy > x)/(N-1)

    out = {} # declares a dictionary to contains the outputs, instead of MATLAB struct

    out['pnn5'] = PNNxfn(5) # 0.0055*sigma
    out['pnn10'] = PNNxfn(10)
    out['pnn20'] = PNNxfn(20)
    out['pnn30'] = PNNxfn(30)
    out['pnn40'] = PNNxfn(40)

    #calculate PSD, DOES NOT MATCH UP WITH MATLAB -----------------------------------------------------------------
    F, Pxx = signal.periodogram(y, window= np.hanning(N)) #hanning confirmed to do the same thing as hann in matlab, periodogram() is what differs

    # calculate spectral measures such as subband spectral power percentage, LF/HF ratio etc.


    LF_lo = 0.04
    LF_hi = 0.15
    HF_lo = 0.15
    HF_hi = 0.4

    fbinsize = F[1] - F[0]

    #calculating indl, indh, indv; needed for loop for python implementation
    indl = []
    for x in F:
        if x >= LF_lo and x <= LF_hi:
            indl.append(1)
        else :
            indl.append(0)


    indh = []
    for x in F:
        if x >= HF_lo and x <= HF_hi:
            indh.append(1)
        else:
            indh.append(0)
    #print("indh: ", indh)

    indv = []
    for x in F:
        if x <= LF_lo:
            indv.append(1)
        else :
            indv.append(0)
    #print("indv: ", indv)

    #calculating lfp, hfp, and vlfp, needed for loop for python implementation
    indlPxx = []
    for i in range(0, len(Pxx)):
        if indl[i] == 1:
            indlPxx.append(Pxx[i])
    lfp = fbinsize * np.sum(indlPxx)
    #print()
    #print('lfp: ', lfp)

    indhPxx = []
    for i in range(0, len(Pxx)):
        if indh[i] == 1:
            indhPxx.append(Pxx[i])
    hfp = fbinsize * np.sum(indhPxx)
    #print('hfp: ', hfp)

    indvPxx = []
    for i in range(0, len(Pxx)):
        if indv[i] == 1:
            indvPxx.append(Pxx[i])
    vlfp = fbinsize * np.sum(indvPxx)
    #print('vlfp: ', vlfp)

    out['lfhf'] = lfp / hfp
    total = fbinsize * np.sum(Pxx)
    out['vlf'] = vlfp/total * 100
    out['lf'] = lfp/total * 100
    out['hf'] = hfp/total * 100


    # triangular histogram index: ----------------------------------------------------------------------
    numBins = 10
    hist = np.histogram(y, bins=numBins)
    out['tri'] = len(y)/np.max(hist[0])


    # Poincare plot measures ---------------------------------------------------------------------------
    rmssd = np.std(diffy, ddof=1) #set delta degrees of freedom to 1 to get same result as matlab
    sigma = np.std(y, ddof=1)

    out["SD1"] = 1/math.sqrt(2) * rmssd * 1000
    out["SD2"] = math.sqrt(2 * sigma**2 - (1/2) * rmssd**2) * 1000

    return out
    # Anonymous function to do the PNNx calculation
    # proportion of the difference magnitudes greater than X*sigma

#@numba.jit(nopython=True,parallel=True)
def DN_STD(y):
    #y must be numpy array
    # if not isinstance(y,np.ndarray):
    #     y = np.asarray(y)
    return(np.std(y))


import numpy as np

def MD_pNN(x):
    """
    pNNx measures of heart rate variability

    Applies pNNx measures to the time series assumed
    to represent sequences of consecutive RR intervals
    measured in milliseconds

    This code is heavily derived from MD_hrv_classic.m because
    it doesn't make medical sense to do a PNN on a z-scored time series.
    But now PSD doesn't make too much sense, so we just evaluate the pNN
    measures.

    :param x: the input time series
    :return: pNNx percentages in a dict
    """

    # Standard defaults --------------------------------
    diffx = np.diff(x)
    N = len(x)

    # Calculate pNNx percentage ------------------------

    Dx = np.abs(diffx) * 1000 # assume milliseconds as for RR intervals
    pnns = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    out = {} # dict used for output in place of MATLAB struct

    for x in pnns:
        out["pnn" + str(x) ] = sum(Dx > x) / (N-1)

    return out






import numpy as np
from scipy import stats
import statsmodels.sandbox.stats.runs as runs

# 18/21 output statistics fully implemented from MATLAB, the other three are either from complex helper functions or MATLAB functions that don't transfer well

def PH_Walker(y, walkerRule='prop', walkerParams=np.array([])):
    """

    PH_Walker simulates a hypothetical walker moving through the time domain

    the hypothetical particle (or 'walker') moves in response to values of the time series at each point

    Outputs from this operation are summaries of the walkers motion, and comparisons of it to the original time series

    :param y: the input time series
    :param walkerRule: the kinematic rule by which the walker moves in response to the time series over time
            (i) 'prop': the walker narrows the gap between its value and that of the time series by a given proportion p

            (ii) 'biasprop': the walker is biased to move more in one direction; when it is being pushed up by the time
            series, it narrows the gap by a proportion p_{up}, and when it is being pushed down by the
            time series it narrows the gap by a (potentially different) proportion p_{down}. walkerParams = [pup,pdown]

            (iii) 'momentum': the walker moves as if it has mass m and inertia
             from the previous time step and the time series acts
             as a force altering its motion in a classical
             Newtonian dynamics framework. [walkerParams = m], the mass.

             (iv) 'runningvar': the walker moves with inertia as above, but
             its values are also adjusted so as to match the local
             variance of time series by a multiplicative factor.
             walkerParams = [m,wl], where m is the inertial mass and wl
             is the window length.

    :param walkerParams: the parameters for the specified walker, explained above

    :return: include the mean, spread, maximum, minimum, and autocorrelation of
            the walker's trajectory, the number of crossings between the walker and the
            original time series, the ratio or difference of some basic summary statistics
            between the original time series and the walker, an Ansari-Bradley test
            comparing the distributions of the walker and original time series, and
            various statistics summarizing properties of the residuals between the
            walker's trajectory and the original time series.

    """

    # ----------------------------------------------------------------------------------------------------------------------------------
    # PRELIMINARIES
    #----------------------------------------------------------------------------------------------------------------------------------

    N = len(y)

    #----------------------------------------------------------------------------------------------------------------------------------
    # CHECK INPUTS
    #----------------------------------------------------------------------------------------------------------------------------------
    if walkerRule == 'runningvar':
        walkerParams = [1.5, 50]
    if (len(walkerParams) == 0):

        if walkerRule == 'prop':
            walkerParams = np.array([0.5])
        if walkerRule == 'biasprop':
            walkerParams = np.array([0.1, 0.2])
        if walkerRule == 'momentum':
            walkerParams = np.array([2])
        if walkerRule == 'runningvar':
            walkerParams = [1.5, 50]

    #----------------------------------------------------------------------------------------------------------------------------------
    # (1) WALK
    #----------------------------------------------------------------------------------------------------------------------------------


    w = np.zeros(N)

    if walkerRule == 'prop':

        # walker starts at zero and narrows the gap between its position
        # and the time series value at that point by the proportion given
        # in walkerParams, to give the value at the subsequent time step
        if isinstance(walkerParams,list):
            walkerParams = walkerParams[0]
        p = walkerParams
        w[0] = 0

        for i in range(1, N):
            w[i] = w[i-1] + p*(y[i-1]-w[i-1])


    elif walkerRule == 'biasprop':
        # walker is biased in one or the other direction (i.e., prefers to
        # go up, or down). Requires a vector of inputs: [p_up, p_down]

        pup = walkerParams[0]
        pdown = walkerParams[0]

        w[0] = 0

        for i in range (1, N):
            if y[i] > y[i-1]:
                w[i] = w[i-1] + pup*(y[i-1]-w[i-1])

            else :
                w[i] = w[i-1] + pdown*(y[i-1]-w[i-1])

    elif walkerRule == 'momentum':
        # walker moves as if it had inertia from the previous time step,
        # i.e., it 'wants' to move the same amount; the time series acts as
        # a force changing its motion

        m = walkerParams[0] # inertial mass

        w[0] = y[0]
        w[1] = y[1]

        for i in range(2, N):
            w_inert = w[i-1] + (w[i-1]-w[i-2])
            w[i] = w_inert + (y[i] - w_inert)/m # dissipative term
            #equation of motion (s-s_0 = ut + F/m*t^2)
            #where the 'force' is F is the change in the original time series at the point

    elif walkerRule == 'runningvar':

        m = walkerParams[0]
        wl = walkerParams[1]

        w[0] = y[0]
        w[1] = y[1]

        for i in range(2, N):
            w_inert = w[i-1] + (w[i-1]-w[i-2])
            w_mom = w_inert + (y[i] - w_inert)/m #dissipative term from time series

            if i > wl:
                w[i] = w_mom * (np.std(y[(i-wl):i]))/np.std(w[(i-wl):i])

            else:
                w[i] = w_mom


    else :

        print("Error: Unknown method: " + walkerRule + " for simulating walker on the time series")


    #----------------------------------------------------------------------------------------------------------------------------------
    # (2) STATISITICS ON THE WALK
    #----------------------------------------------------------------------------------------------------------------------------------

    out = {} # dictionary for storing variables

    # (i) The walk itself -------------------------------------------------------------------------------------------

    out['w_mean'] = np.mean(w)
    out['w_median'] = np.median(w)
    out['w_std'] = np.std(w)
    out['w_ac1'] = CO_AutoCorr(w, 1, method='timedomainstat') # this function call in MATLAB uses method='Fourier', but we don't have that case implemented yet in autoCorr, however this seems to output the same thing
    out['w_ac2'] = CO_AutoCorr(w, 2, method='timedomainstat')
    out['w_tau'] = CO_FirstZero(w, 'ac')
    out['w_min'] = np.min(w)
    out['w_max'] = np.max(w)
    out['propzcross'] = sum( np.multiply( w[0:(len(w)-2)], w[1:(len(w)-1)] ) < 0) / (N-1) # np.multiply performs elementwise multiplication like matlab .*
    # differences between the walk at signal

    # (ii) Differences between the walk at signal -------------------------------------------------------------------

    out['sw_meanabsdiff'] = np.mean(np.abs(y-w))
    out['sw_taudiff'] = CO_FirstZero(y, 'ac') - CO_FirstZero(w, 'ac')
    out['sw_stdrat'] = np.std(w)/np.std(y) # will be thse same as w_std for z-scored signal
    out['sw_ac1rat'] = out['w_ac1']/CO_AutoCorr(y, 1)
    out['sw_minrat'] = min(w)/min(y)
    out['sw_maxrat'] = max(w)/max(y)
    out['sw_propcross'] = sum(np.multiply( w[0:(len(w)-1)] - y[0:(len(y)-1)] , w[1:(len(w))]-y[1:(len(y))]) < 0 )/(N-1) #np.multiply performs elementwise multiplication like matlab .*

    ansari = stats.ansari(w, y)
    out['sw_ansarib_pval'] = ansari[1]


    # r = np.linspace( np.min(np.min(y), np.min(w)), np.max(np.max(y), np.max(w)), 200 )
    # dy = stats.gaussian_kde(y, r)


    # (iii) looking at residuals between time series and walker

    res = w-y

    # CLOSEST FUNCTION TO MATLAB RUNSTEST, found in statsmodels.sandbox.stats.runs
    # runstest = runs.runstest_2samp(res, groups=2)
    # out['res_runstest'] = runstest

    out['res_acl'] = CO_AutoCorr(res, lag=1)


    return out

def CO_trev(y,tau = 'ac'):

    if tau == 'ac':

        tau = CO_FirstZero(y,'ac')

    # else:
    #
    #     tau = CO_FirstMin(y,'mi')

    N = len(y)

    yn = y[0:N-tau]
    yn1 = y[tau:N]

    try:
        raw = np.mean(np.power(yn1-yn,3)) / np.mean(np.power(yn1 - yn,2))**(3/2)
    except:
        return {'raw':np.nan,'abs':np.nan,'num':np.nan,'absnum':np.nan,'denom':np.nan}

    outDict = {}

    outDict['raw'] = raw

    outDict['abs'] = np.absolute(raw)

    outDict['num'] = np.mean(np.power(yn1-yn,3))

    outDict['absnum'] = np.absolute(outDict['num'])

    outDict['denom'] = np.mean(np.power(yn1-yn,2))**(3/2)

    return outDict

def SY_LocalGlobal(y,subsetHow = 'l',n = ''):

    if subsetHow == 'p' and n == '':

        n = .1

    elif n == '':

        n = 100

    N  = len(y)

    if subsetHow == 'l':

        r = range(0,min(n,N))

    elif subsetHow == 'p':

        if n > 1:

            n = .1

        r = range(0,round(N*n))

    elif subsetHow == 'unicg':

        r = np.round(np.arange(0,N,n)).astype(int)

    elif subsetHow == 'randcg':

        r = np.random.randint(N,size = n)

    if len(r)<5:

        out = np.nan
        return out

    out = {}

    out['absmean'] = np.absolute(np.mean(y[r]))
    out['std'] = np.std(y[r],ddof = 1)
    out['median'] = np.median(y[r])
    out['iqr'] = np.absolute(( 1 - stats.iqr(y[r]) /stats.iqr(y) ))

    if stats.skew(y) == 0:

        out['skew'] = np.nan

    else:

        out['skew'] = np.absolute( 1 - stats.skew(y[r]) / stats.skew(y) )

    out['kurtosis'] = np.absolute(1 - stats.kurtosis(y[r],fisher=False) / stats.kurtosis(y,fisher=False))

    out['ac1'] = np.absolute( 1- CO_AutoCorr(y[r],1,'Fourier') / CO_AutoCorr(y,1,'Fourier'))


    return out

import itertools
#import numba

#@numba.jit(nopython=True,parallel=True)
def EN_PermEn(y,m = 2,tau = 1):

    try:

        x = BF_embed(y,tau,m)

    except:

        return np.nan



    Nx = x.shape[0]

    permList = perms(m)
    numPerms = len(permList)


    countPerms = np.zeros(numPerms)

    for j in range(Nx):

        ix = np.argsort(x[j,:])

        for k in range(numPerms):


            if (permList[k,:] - ix == np.zeros(m)).all() :

                countPerms[k] = countPerms[k] + 1

                break

    p = countPerms / Nx

    p_0 = p[p > 0]

    permEn = -sum(np.multiply(p_0,np.log2(p_0)))

    mFact = math.factorial(m)

    normPermEn = permEn / np.log2(mFact)

    p_LE = np.maximum(np.repeat(1 / Nx,p.shape),p)

    permENLE = - np.sum(np.multiply(p_LE,np.log(p_LE))) / (m - 1)

    out = {'permEn':permEn,'normPermEn':normPermEn,'permEnLE':permENLE}

    return out

def perms(n):

    permut = itertools.permutations(np.arange(n))

    permut_array = np.empty((0,n))

    for p in permut:

        permut_array = np.append(permut_array,np.atleast_2d(p),axis=0)

    return(permut_array)

#import warnings
#@numba.jit(nopython=True,parallel=True)
def DN_cv(x,k = 1):
    # if k % 1 != 0 or k < 0:
    #     warnings.warn("k should probably be positive int")
    if np.mean(x) == 0:
        return np.nan
    return (np.std(x)**k) / (np.mean(x)**k)

def SB_BinaryStats(y,binaryMethod = 'diff'):

    yBin = BF_Binarize(y,binaryMethod)

    N = len(yBin)

    outDict = {}

    outDict['pupstat2'] = np.sum((yBin[math.floor(N /2):] == 1))  / np.sum((yBin[:math.floor(N /2)] == 1))

    stretch1 = []
    stretch0 = []
    count = 1



    for i in range(1,N):

        if yBin[i] == yBin[i - 1]:

            count = count + 1

        else:

            if yBin[i - 1] == 1:

                stretch1.append(count)

            else:

                stretch0.append(count)
            count = 1
    if yBin[N-1] == 1:

        stretch1.append(count)

    else:

        stretch0.append(count)


    outDict['pstretch1'] = len(stretch1) / N

    if stretch0 == []:

        outDict['longstretch0'] = 0
        outDict['meanstretch0'] = 0
        outDict['stdstretch0'] = None

    else:

        outDict['longstretch0'] = np.max(stretch0)
        outDict['meanstretch0'] = np.mean(stretch0)
        outDict['stdstretch0'] = np.std(stretch0,ddof = 1)

    if stretch1 == []:

        outDict['longstretch1'] = 0
        outDict['meanstretch1'] = 0
        outDict['stdstretch1'] = None

    else:

        outDict['longstretch1'] = np.max(stretch1)
        outDict['meanstretch1'] = np.mean(stretch1)
        outDict['stdstretch1'] = np.std(stretch1,ddof = 1)

    try:

        outDict['meanstretchdiff'] = outDict['meanstretch1'] - outDict['meanstretch0']
        outDict['stdstretchdiff'] = outDict['stdstretch1'] - outDict['stdstretch0']

    except:

        pass


    return outDict

def SY_SlidingWindow(y,windowStat = 'mean',acrossWinStat='std',numSeg=5,incMove=2):

    winLength = math.floor(len(y) / numSeg)

    if winLength == 0:

        return

    inc = math.floor(winLength / incMove)

    if inc == 0:

        inc = 1


    numSteps = (math.floor((len(y)-winLength)/inc) + 1)

    qs = np.zeros(numSteps)

    if windowStat == 'mean':

        for i in range(numSteps):

            qs[i] = np.mean(y[getWindow(i)])

    elif windowStat == 'std':

        for i in range(numSteps):

            qs[i] = np.std(y[getWindow(i)])


    elif windowStat == 'apen':

        for i in range(numSteps):

            qs[i] = EN_ApEn(y[getWindow(i)],1,.2)

    elif windowStat == 'sampen':

        for i in range(numSteps):

            sampStruct = EN_SampEn(y[getWindow(i)],1,.1)
            qs[i] = sampStruct['Sample Entropy']

    elif windowStat == 'mom3':

        for i in range(numSteps):

            qs[i] = DN_Moments(y[getWindow(i)],3)

    elif windowStat == 'mom4':

        for i in range(numSteps):

            qs[i] = DN_Moments(y[getWindow(i)],4)

    elif windowStat == 'mom5':

        for i in range(numSteps):

            qs[i] = DN_Moments(y[getWindow(i)],5)

    elif windowStat == 'AC1':

        for i in range(numSteps):

            qs[i] = CO_AutoCorr(y[getWindow(i)],1,'Fourier')


    if acrossWinStat == 'std':

        out = np.std(qs)  / np.std(y)

    elif acrossWinStat == 'apen':

        out = EN_ApEn(qs,2,.15)

    elif acrossWinStat == 'sampen':

        out = EN_SampEn(qs,2,.15)['Sample Entropy']

    else:

        out = None

    return out



def getWindow(stepInd,inc,winLength):

    return np.arange((stepInd - 1)*inc,(stepInd - 1)*inc + winLength)

def SB_MotifThree( y, cgHow = 'quantile'):

    numLetters = 3

    if cgHow == 'quantile':

        yt = SB_CoarseGrain(y,'quantile',numLetters)

    elif cgHow == 'diffquant':

        yt = SB_CoarseGrain(np.diff(y),'quantile',numLetters)

    else:

        return


    N = len(yt)


    r1 = [[],[],[]]

    out1 = np.zeros(3)

    for i in range(1,4):

        r1[i - 1] = np.argwhere(yt == i)[:,0]

        out1[i - 1] = len(r1[i - 1]) / N

    outDict = {}

    outDict['a'] = out1[0]
    outDict['b'] = out1[1]
    outDict['c'] = out1[2]
    outDict['h'] = f_entropy(out1)

    for i in range(3):

        if len(r1[i]) == 0:

            continue

        if r1[i][-1] == N - 1:

            r1[i] = r1[i][:-1]

    r2 =[[[],[],[]],[[],[],[]],[[],[],[]]]



    out2 = np.zeros((3,3))

    for i in range(1,4):

        iIndex = i - 1

        for j in range(1,4):

            jIndex = j - 1

            r2[iIndex][jIndex] = r1[iIndex][np.argwhere(yt[ r1[iIndex] + 1 ] == j)][:,0]

            out2[iIndex,jIndex] = len(r2[iIndex][jIndex]) / (N-1)

    for i in range(3):

        for j in range(3):

            if len(r2[i][j]) == 0:

                continue

            if r2[i][j][-1] == N - 2:

                r2[i][j] = r2[i][j][:-1]





    r3 =[[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]]

    out3 = np.zeros((3,3,3))

    for i in range(1,4):

        iIndex = i - 1

        for j in range(1,4):

            jIndex = j - 1

            for k in range(1,4):

                kIndex = k -1

                r3[iIndex][jIndex][kIndex] = r2[iIndex][jIndex][np.argwhere(yt[ r2[iIndex][jIndex] + 2 ] == k)][:,0]

                out3[iIndex,jIndex,kIndex] = len(r3[iIndex][jIndex][kIndex]) / (N-2)


    letters = ['a','b','c']


    for i in range(3):

        l1 = letters[i]

        for j in range(3):

            l2 = letters[j]

            outDict[l1 + l2] = out2[i,j]

    outDict['hh'] = f_entropy(out2)

    for i in range(3):

        l1 = letters[i]

        for j in range(3):

            l2 = letters[j]

            for k in range(3):

                l3 = letters[k]

                outDict[l1 + l2 + l3] = out3[i,j,k]

    outDict['hhh'] = f_entropy(out3)

    #Seems very ineffiecnt probs can use other numpy functions to speed up
    for i in range(3):

        for j in range(3):

            for k in range(3):

                if len(r3[i][j][k]) == 0:

                    continue


                if r3[i][j][k][-1] == N - 3:

                    r3[i][j][k] = r3[i][j][k][:-1]

    r4 =[[[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]],[[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]],[[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]]]

    out4 = np.zeros((3,3,3,3))

    for i in range(1,4):

        iIndex = i - 1

        for j in range(1,4):

            jIndex = j - 1

            for k in range(1,4):

                kIndex = k -1

                for l in range(1,4):

                    lIndex = l - 1

                    r4[iIndex][jIndex][kIndex][lIndex] = r3[iIndex][jIndex][kIndex][np.argwhere(yt[ r3[iIndex][jIndex][kIndex] + 3 ] == l)][:,0]

                    out4[iIndex,jIndex,kIndex,lIndex] = len(r4[iIndex][jIndex][kIndex][lIndex]) / (N-3)
    for i in range(3):

        l1 = letters[i]

        for j in range(3):

            l2 = letters[j]

            for k in range(3):

                l3 = letters[k]

                for l in range(3):

                    l4 = letters[l]

                    outDict[l1 + l2 + l3 + l4] = out4[i,j,k,l]

    outDict['hhhh'] = f_entropy(out4)


    return outDict


def f_entropy(x):

    h = -np.sum(np.multiply(x[ x > 0],np.log(x[x > 0])))

    return h

#@numba.jit(nopython=True,parallel=True)
def DN_TrimmedMean(y,n = 0):
    N = len(y)
    trim = int(np.round(N * n / 2))
    y = np.sort(y)
    #return stats.trim_mean(y,n) doesn't agree with matlab
    return np.mean(y[trim:N-trim])

def ST_LocalExtrema(y,lorf = 'l',n = ''):
    if lorf == 'l' and n == '':
        n = 100
    elif n == '':
        n = 5

    N = len(y)

    if lorf == 'l':
        wl = n
    elif lorf == 'n':
        wl = math.floor(N/n)
    else:
        wl = CO_FirstZero(y,'ac')

    if wl > N or wl <= 1:
        #print('window too short or long')
        return np.nan

    y_buffer = BF_makeBuffer(y,wl).transpose()

    numWindows = y_buffer.shape[1]

    locmax = np.max(y_buffer,axis = 0)

    locmin = np.min(y_buffer,axis = 0)

    abslocmin = np.absolute(locmin)

    exti = np.where(abslocmin > locmax)

    locext = locmax

    locext[exti] = locmin[exti]

    abslocext = np.absolute(locext)

    out = {}

    out['meanrat'] = np.mean(locmax)/np.mean(abslocmin)
    out['medianrat'] = np.median(locmax)/np.median(abslocmin)
    out['minmax'] = np.min(locmax)
    out['minabsmin'] = np.min(abslocmin)
    out['minmaxonminabsmin'] = np.min(locmax)/np.min(abslocmin)
    out['meanmax'] = np.mean(locmax)
    out['meanabsmin'] = np.mean(abslocmin)
    out['meanext'] = np.mean(locext)
    out['medianmax'] = np.median(locmax)
    out['medianabsmin'] = np.median(abslocmin)
    out['medianext'] = np.median(locext)
    out['stdmax'] = np.std(locmax,ddof=1)
    out['stdmin'] = np.std(locmin,ddof=1)
    out['stdext'] = np.std(locext,ddof=1)
    #out.zcext = ST_SimpleStats(locext,'zcross');
    out['meanabsext'] = np.mean(abslocext)
    out['medianabsext'] = np.median(abslocext)
    out['diffmaxabsmin'] = np.sum(np.absolute(locmax-abslocmin))/numWindows
    out['uord'] = np.sum(np.sign(locext))/numWindows #% whether extreme events are more up or down
    out['maxmaxmed'] = np.max(locmax)/np.median(locmax)
    out['minminmed'] = np.min(locmin)/np.median(locmin)
    out['maxabsext'] = np.max(abslocext)/np.median(abslocext)

    return out

def MF_ARMA_orders(y,pr = [1,2,3,4,5],qr=[0,1,2,3,4,5]):

    y = (y - np.mean(y)) / np.std(y)

    aics = np.zeros((len(pr),len(qr)))
    bics = np.zeros((len(pr),len(qr)))

    for i in range(len(pr)):

        for j in range(len(qr)):

            p = pr[i]
            q = qr[i]

            try:

                model = ARIMA(y, order=(p,0,q))
                model_fit = model.fit( disp=False)

            except:
                print("FAILED ARMA MODEL")
                return None
            aics[i,j] = model_fit.aic
            bics[i,j] = model_fit.bic

    outDict = {}

    outDict['aic_min'] = np.min(aics)

    mins = np.argwhere(aics == np.min(aics))[0]

    outDict['opt_p'] = pr[mins[0]]

    outDict['opt_q'] = qr[mins[0]]

    outDict['meanAICS'] = np.mean(aics)
    outDict['stdAICS'] = np.std(aics)

    outDict['meanBICS'] = np.mean(bics)
    outDict['stdBICS'] = np.std(bics)

    return outDict

def CO_Embed2_Basic(y,tau = 5,scale = 1):
    '''
    CO_Embed2_Basic Point density statistics in a 2-d embedding space
    %
    % Computes a set of point density measures in a plot of y_i against y_{i-tau}.
    %
    % INPUTS:
    % y, the input time series
    %
    % tau, the time lag (can be set to 'tau' to set the time lag the first zero
    %                       crossing of the autocorrelation function)
    %scale since .1 and .5 don't make sense for HR RESP ...
    % Outputs include the number of points near the diagonal, and similarly, the
    % number of points that are close to certain geometric shapes in the y_{i-tau},
    % y_{tau} plot, including parabolas, rings, and circles.
    '''
    if tau == 'tau':

        tau = CO_FirstZero(y,'ac')

    xt = y[:-tau]
    xtp = y[tau:]
    N = len(y) - tau

    outDict = {}

    outDict['updiag01'] = np.sum((np.absolute(xtp - xt) < 1*scale)) / N
    outDict['updiag05'] = np.sum((np.absolute(xtp - xt) < 5*scale)) / N

    outDict['downdiag01'] = np.sum((np.absolute(xtp + xt) < 1*scale)) / N
    outDict['downdiag05'] = np.sum((np.absolute(xtp + xt) < 5*scale)) / N

    outDict['ratdiag01'] = outDict['updiag01'] / outDict['downdiag01']
    outDict['ratdiag05'] = outDict['updiag05'] / outDict['downdiag05']

    outDict['parabup01'] = np.sum( ( np.absolute( xtp - np.square(xt) ) < 1 * scale ) ) / N
    outDict['parabup05'] = np.sum( ( np.absolute( xtp - np.square(xt) ) < 5 * scale ) ) / N

    outDict['parabdown01'] = np.sum( ( np.absolute( xtp + np.square(xt) ) < 1 * scale ) ) / N
    outDict['parabdown05'] = np.sum( ( np.absolute( xtp + np.square(xt) ) < 5 * scale ) ) / N

    outDict['parabup01_1'] = np.sum( ( np.absolute( xtp -( np.square(xt) + 1 ) ) < 1 * scale ) ) / N
    outDict['parabup05_1'] = np.sum( ( np.absolute( xtp - (np.square(xt) + 1)) < 5 * scale ) ) / N

    outDict['parabdown01_1'] = np.sum( ( np.absolute( xtp + np.square(xt) - 1 ) < 1 * scale ) ) / N
    outDict['parabdown05_1'] = np.sum( ( np.absolute( xtp + np.square(xt) - 1) < 5 * scale ) ) / N

    outDict['parabup01_n1'] = np.sum( ( np.absolute( xtp -( np.square(xt) - 1 ) ) < 1 * scale ) ) / N
    outDict['parabup05_n1'] = np.sum( ( np.absolute( xtp - (np.square(xt) - 1)) < 5 * scale ) ) / N

    outDict['parabdown01_n1'] = np.sum( ( np.absolute( xtp + np.square(xt) + 1 ) < 1 * scale ) ) / N
    outDict['parabdown05_n1'] = np.sum( ( np.absolute( xtp + np.square(xt) + 1) < 5 * scale ) ) / N

    outDict['ring1_01'] = np.sum( np.absolute( np.square(xtp) + np.square(xt) - 1 ) < 1 * scale ) / N
    outDict['ring1_02'] = np.sum( np.absolute( np.square(xtp) + np.square(xt) - 1 ) < 2 * scale ) / N
    outDict['ring1_05'] = np.sum( np.absolute( np.square(xtp) + np.square(xt) - 1 ) < 5 * scale ) / N

    outDict['incircle_01'] = np.sum(  np.square(xtp) + np.square(xt)  < 1 * scale ) / N
    outDict['incircle_02'] = np.sum( np.square(xtp) + np.square(xt)  < 2 * scale ) / N
    outDict['incircle_05'] = np.sum(  np.square(xtp) + np.square(xt)  < 5 * scale ) / N


    outDict['incircle_1'] = np.sum(  np.square(xtp) + np.square(xt)  < 10 * scale ) / N
    outDict['incircle_2'] = np.sum( np.square(xtp) + np.square(xt)  < 20 * scale ) / N
    outDict['incircle_3'] = np.sum(  np.square(xtp) + np.square(xt)  < 30 * scale ) / N

    outDict['medianincircle'] = np.median([outDict['incircle_01'], outDict['incircle_02'], outDict['incircle_05'], \
                            outDict['incircle_1'], outDict['incircle_2'], outDict['incircle_3']])

    outDict['stdincircle'] = np.std([outDict['incircle_01'], outDict['incircle_02'], outDict['incircle_05'], \
                            outDict['incircle_1'], outDict['incircle_2'], outDict['incircle_3']],ddof = 1)


    return outDict

def SC_DFA(y):

    N = len(y)

    tau = int(np.floor(N/2))

    y = y - np.mean(y)

    x = np.cumsum(y)

    taus = np.arange(5,tau+1)

    ntau = len(taus)

    F = np.zeros(ntau)

    for i in range(ntau):

        t = int(taus[i])



        x_buff = x[:N - N % t]

        x_buff = x_buff.reshape((int(N / t),t))


        y_buff = np.zeros((int(N / t),t))

        for j in range(int(N / t)):

            tt = range(0,int(t))

            p = np.polyfit(tt,x_buff[j,:],1)

            y_buff[j,:] =  np.power(x_buff[j,:] - np.polyval(p,tt),2)



        y_buff.reshape((N - N % t,1))

        F[i] = np.sqrt(np.mean(y_buff))

    logtaur = np.log(taus)

    logF = np.log(F)

    p = np.polyfit(logtaur,logF,1)

    return p[0]

#@numba.jit(nopython=True,parallel=True)
def DN_HighLowMu(y):
    mu = np.mean(y)
    mhi = np.mean(y[y>mu])
    mlo = np.mean(y[y<mu])
    return (mhi - mu) / (mu - mlo)
