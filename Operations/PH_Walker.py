
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
