
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




