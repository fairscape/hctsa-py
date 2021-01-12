#© 2020 By The Rector And Visitors Of The University Of Virginia

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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
