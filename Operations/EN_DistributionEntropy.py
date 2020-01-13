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
