
import numpy as np
import Operations
import scipy as sc
from scipy import stats


def SB_CoarseGrain(y, howtocg, numGroups):
    '''
    Coarse-grains a continuous time series to a discrete alphabet

    ------ Inputs:

    y1          : the continuous time series

    howtocg     : the method of course-graining

    numGroups   : either specifies the size of the alphabet for 'quantile' and 'updown' or sets the time delay for
        the embedding subroutines
    '''

    # --------------------------------------------------------------------------------------------------------------------------------
    # CHECK INPUTS, SET UP PRELIMINARIES
    # --------------------------------------------------------------------------------------------------------------------------------

    N = len(y)
    
    if howtocg not in ['quantile', 'updown', 'embed2quadrants', 'embed2octants']:
        print("Error: "+ howtocg + " is an unknown coarse-graining method")
        print("Choose between: quantile, updown, embed2quadrants, embed2octants ")
        return

    # some course-graining/symbolization methods require initial processing ---------------------------
        #Python does not have a switch-case like matlab, so if else statements were used instead

    if howtocg == 'updown':

        y = np.diff(y)
        N = N - 1 # the time series is one value shorter than the input because of differenceing
        howtocg = 'quantile'

    elif howtocg == 'embed2quadrants' or 'embed2octants':
        # construct the embedding

        if (numGroups == 'tau'):
            tau = CO_FirstZero(y, 'ac') #first zero crossing of the autocorrelation data
        else:
            tau = numGroups

        if tau > N/25:
            tau = int(np.floor(N/25))

        m1 = y[0:(N-tau)]
        m2 = y[tau:N]

        upr = np.argwhere(m2 >= 0)
        downr = np.argwhere(m2 < 0)

        q1r = upr[m1[upr] >= 0]
        q2r = upr[m1[upr] < 0]
        q3r = downr[m1[downr] < 0]
        q4r = downr[m1[downr] >= 0]


    # ----------------------------------------------------------------------------------------------------------------------------
    # CARRY OUT THE COARSE GRAINING
    # ----------------------------------------------------------------------------------------------------------------------------

    if howtocg == 'quantile':

        th = sc.stats.mstats.mquantiles(y, np.linspace(0, 1, numGroups + 1), alphap=0.5, betap=0.5) # matlab uses peicewise linear interpolation for calculating quantiles, hence alphap=0.5, betap=0.5
        th[0] = th[0] - 1 # ensures the first point is included
        yth = np.zeros([N, 1])

        for i in range(0, numGroups):  # had some trouble here finding indexes that satisfy two conditions in python, hence this is written differently then in matlab

            z = ((y > th[i]) & (y <= th[i+1])).nonzero()[0]

            for x in range(0, len(z)):
                yth[z[x]] = i + 1

    elif howtocg == 'embed2quadrants':
        #turn the time series data into a set of numbers from 1:numGroups

        yth = np.zeros([len(m1), 1])

        yth[q1r] = 1
        yth[q2r] = 2
        yth[q3r] = 3
        yth[q4r] = 4

    elif howtocg =='embed2octants':

        #devide based on octants in 2D embedding space
        o1r = q1r[m2[q1r] < m1[q1r]]
        o2r = q1r[m2[q1r] >= m1[q1r]]
        o3r = q2r[m2[q2r] >= -m1[q2r]]
        o4r = q2r[m2[q2r] < -m1[q2r]]
        o5r = q3r[m2[q3r] >= m1[q3r]]
        o6r = q3r[m2[q3r] < m1[q3r]]
        o7r = q4r[m2[q4r] < -m1[q4r]]
        o8r = q4r[m2[q4r] >= -m1[q4r]]

        yth = np.zeros([len(m1),1])
        yth[o1r] = 1
        yth[o2r] = 2
        yth[o3r] = 3
        yth[o4r] = 4
        yth[o5r] = 5
        yth[o6r] = 6
        yth[o7r] = 7
        yth[o8r] = 8


    if np.any(yth == 0):
        print(yth)
        print("Error: all values in the sequence were not assigned to a group")

    return yth # return the output
