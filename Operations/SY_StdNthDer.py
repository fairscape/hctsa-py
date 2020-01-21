
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
