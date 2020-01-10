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

    outDict['updiag1'] = np.sum((np.absolute(xtp - xt) < 1*scale)) / N
    outDict['updiag5'] = np.sum((np.absolute(xtp - xt) < 5*scale)) / N

    outDict['downdiag1'] = np.sum((np.absolute(xtp + xt) < 1*scale)) / N
    outDict['downdiag5'] = np.sum((np.absolute(xtp + xt) < 5*scale)) / N

    outDict['ratdiag1'] = outDict['updiag1'] / outDict['downdiag1']
    outDict['ratdiag5'] = outDict['updiag5'] / outDict['downdiag5']

    outDict['parabup1'] = np.sum( ( np.absolute( xtp - np.square(xt) ) < 1 * scale ) ) / N
    outDict['parabup5'] = np.sum( ( np.absolute( xtp - np.square(xt) ) < 5 * scale ) ) / N

    outDict['parabdown1'] = np.sum( ( np.absolute( xtp + np.square(xt) ) < 1 * scale ) ) / N
    outDict['parabdown5'] = np.sum( ( np.absolute( xtp + np.square(xt) ) < 5 * scale ) ) / N


    return outDict
