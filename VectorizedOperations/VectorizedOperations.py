import numpy as np
import VectorizedOperations.VectorizedOperations as vect
from scipy import stats





def DN_RemovePointsVect(y,removeHow = 'absfar',p = .1,acf_y = None,FirstZero = None):


    if removeHow == 'absclose':

        i =  np.argsort(-np.absolute(y),kind = 'mergesort',axis = 1)

    elif removeHow == 'absfar':

        i = np.argsort(np.absolute(y),kind = 'mergesort',axis = 1)

    elif removeHow == 'min':

        i =  np.argsort(-y,kind = 'mergesort',axis = 1)

    elif removeHow == 'max':

        i = np.argsort(y,kind = 'mergesort',axis = 1)



    N = y.shape[1]
    points = y.shape[0]

    out = np.zeros((points,13))

    rKeep = np.sort(i[:,0:int(np.round(N*(1-p)))],axis = 1)

    y_trim = np.zeros((rKeep.shape))

    for i in range(points):

        y_trim[i,:] = y[i,rKeep[i,:]]

    if acf_y is None:

        acf_y = SUB_acf(y,8)

    acf_y_trim = SUB_acf(y_trim,8)

    if FirstZero is None:

        FirstZero = CO_FirstZeroVect(y)


    out[:,0] = np.divide(CO_FirstZeroVect(y_trim),FirstZero).flatten()

    out[:,1] = np.divide(acf_y_trim[:,0],acf_y[:,0]).flatten()

    out[:,2] = np.absolute(acf_y_trim[:,0] - acf_y[:,0])

    out[:,3] = np.divide(acf_y_trim[:,1],acf_y[:,1]).flatten()

    out[:,4] = np.absolute(acf_y_trim[:,1] - acf_y[:,1])

    out[:,5] = np.divide(acf_y_trim[:,2],acf_y[:,2]).flatten()

    out[:,6] = np.absolute(acf_y_trim[:,2] - acf_y[:,2])

    out[:,7] = np.sum(np.absolute(acf_y_trim-acf_y),axis = 1)

    out[:,8] = np.mean(y_trim,axis = 1)

    out[:,9] = np.median(y_trim,axis = 1)

    out[:,10] = np.std(y_trim)

    out[:,11] = np.divide(stats.skew(y_trim,axis = 1),stats.skew(y,axis = 1))

    out[:,12] = np.divide(stats.kurtosis(y_trim,axis = 1,fisher=False),stats.kurtosis(y,axis = 1,fisher=False))

    return out


def SUB_acf(x,n):

    acf = np.zeros((x.shape[0],n))

    for i in range(n):

        acf[:,i] = CO_AutoCorrVect(x,i,'Fourier')

    return acf




def CO_FirstZeroVect(y,minWhat = 'ac'):

    acf = vect.CO_AutoCorrVect(y,[],'Fourier')

    N = y.shape[1]

    points = y.shape[0]

    result = np.zeros((points,1))

    for i in range(1,N-1):

        #update result if current if less than before
        #and that rows first min wasnt already found

        less = (acf[:,i] <= 0).reshape((points,1))

        notseen = (result == 0)

        result[np.logical_and(less,notseen)] = i

        if np.logical_not(notseen).all():

            return result

    result[notseen] = N

    return result

def CO_FirstMinVect(y,minWhat = 'ac'):

    acf = vect.CO_AutoCorrVect(y,[],'Fourier')

    N = y.shape[1]

    points = y.shape[0]

    result = np.zeros((points,1))

    for i in range(1,N-1):

        #update result if current if less than before
        #and that rows first min wasnt already found

        less = (acf[:,i] - acf[:,i-1] < 0).reshape((points,1))

        notseen = (result == 0)

        result[np.logical_and(less,notseen)] = i

        if np.logical_not(notseen).all():

            return result

    result[notseen] = N

    return result




def CO_AutoCorrVect(y,lag = 1,method = 'Fourier',t = 1, mean = None,std = None):

    if mean is None:

        mean = np.mean(y,axis = 1)
        mean = mean.reshape((mean.shape[0],1))

    if std is None:

        std = np.std(y,axis = 1)

    def ACFy(tau):

        return np.divide(np.mean( np.multiply((y[:,:-tau] - mean),(y[:,tau:] - mean)) ,axis = 1), np.power(std,2))

    if method == 'TimeDomianStat':

        if lag == []:

            acf = np.zeros((y.shape[1],26))

            acf[:,0] = 1

            for i in range(1,26):

                acf[:,i] = ACFy(i)

            return acf

        return ACFy(lag)

    N = y.shape[1]

    points = y.shape[0]

    nFFT = int(2**(np.ceil(np.log2(N)) + 1))

    F = np.fft.fft( y - mean , nFFT, axis = 1)

    F = np.multiply(F,np.conj(F))

    acf = np.fft.ifft(F,axis = 1)

    acf = np.divide(acf,acf[:,0].reshape((points,1)))

    acf = acf.real

    if lag == []:

        return acf

    return acf[:,lag]







def DN_BurstinessVect(y,mean = None,std = None):

    if mean is None:

        mean = np.mean(y,axis = 1)

    if std is None:

        std = np.std(y,axis = 1)

    r = np.divide(std,mean)

    B = np.divide(( r - 1 ), ( r + 1 ))

    return B

def CO_NonlinearAutocorrVect(y,taus,doAbs='empty'):


    if doAbs == 'empty':

        if len(taus) % 2 == 1:

            doAbs = 0

        else:

            doAbs = 1


    N = int(y.shape[1])

    tmax = np.max(taus)

    nlac = y[:,tmax:N]


    for i in taus:

        nlac = np.multiply(nlac,y[:,tmax - i:N - i])

    if doAbs:

        return np.mean(np.absolute(nlac),axis = 1)

    return np.mean(nlac,axis = 1)
