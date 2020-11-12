
import numpy as np
from scipy import stats
from scipy import signal
from PeripheryFunctions import BF_sgnchange as bf

def ST_SimpleStats(x, whatStat):

    '''
    ST_SimpleStats   Basic statistics about an input time series

    ---INPUTS:
    x, the input time series
    whatStat, the statistic to return:


             (i) 'zcross': the proportionof zero-crossings of the time series
                           (z-scored input thus returns mean-crossings)
             (ii) 'maxima': the proportion of the time series that is a local maximum
             (iii) 'minima': the proportion of the time series that is a local minimum
             (iv) 'pmcross': the ratio of the number of times that the (ideally
                             z-scored) time-series crosses +1 (i.e., 1 standard
                             deviation above the mean) to the number of times
                             that it crosses -1 (i.e., 1 standard deviation below
                             the mean)
             (v) 'zsczcross': the ratio of zero crossings of raw to detrended
                               time series where the raw has zero mean

        :param x: the input time series
        :param whatStat: the statistic to return
        :return: basic statistics about the input time series
    '''

    N = len(x)

    if whatStat == 'zcross':
        # Proportion of zero-crossings of the time series
        # in the case of the z-scored input, crosses its mean
        xch = np.multiply(x[0:len(x)-1],  x[1:(len(x))])
        out = sum(xch < 0)/N

    elif whatStat == 'maxima':
        # proportion of local maxima in the time series
        dx = np.diff(x)
        out = np.sum(np.logical_and(dx[0:(len(dx)-1)] > 0, dx[1:len(dx)] < 0)/(N-1)) # must use np.logical_and() instead of & (MATLAB)

    elif whatStat == 'minima':
        dx = np.diff(x)
        out = np.sum(np.logical_and(dx[0:(len(dx)-1)] < 0, dx[1:len(dx)] > 0)/(N-1)) # must use np.logical_and() instead of & (MATLAB)

    elif whatStat == 'pmcross':
        # ratio of times cross 1 to -1
        c1sig = np.sum(bf.BF_sgnchange(x-1)) # num times cross 1
        c2sig = np.sum(bf.BF_sgnchange(x+1)) # num times cross -1

        if c2sig == 0:
            out = None # NaN in matlab original
        else:
            out = c1sig/c2sig

    elif whatStat == 'zsczcross':
        # ratio of zero crossings of raw to detrended time series
        # where the raw has zero mean
        x = stats.zscore(x)
        xch = np.multiply(x[0:N-1], x[1:N])
        h1 = np.sum( xch < 0 )
        y = signal.detrend(x)
        ych = np.multiply(y[0:len(y)-1], y[1:len(y)])
        h2 = np.sum(ych < 0)

        if h1 == 0:
            out = None # instead of NaN (original MATLAB)
        else:
            out = h2/h1

    else:
        print("---------------- Unknown statistic: " + whatStat + " --------------------")
        print("-- Choose from: zcross, maxima, minima, pmcross, or zsczcross --")
        return

    return out
