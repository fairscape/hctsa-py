def EX_MovingThreshold(y,a = 1,b = .1):
    '''
     EX_MovingThreshold    Moving threshold model for extreme events in a time series

     Inspired by an idea contained in:
     "Reactions to extreme events: Moving threshold model"
     Altmann et al., Physica A 364, 435--444 (2006)

     This algorithm is based on this idea: it uses the occurrence of extreme events
     to modify a hypothetical 'barrier' that classes new points as 'extreme' or not.
     The barrier begins at sigma, and if the absolute value of the next data point
     is greater than the barrier, the barrier is increased by a proportion 'a',
     otherwise the position of the barrier is decreased by a proportion 'b'.

    ---INPUTS:
     y, the input (z-scored) time series
     a, the barrier jump parameter (in extreme event)
     b, the barrier decay proportion (in absence of extreme event)

    ---OUTPUTS: the mean, spread, maximum, and minimum of the time series for the
     barrier, the mean of the difference between the barrier and the time series
     values, and statistics on the occurrence of 'kicks' (times at which the
     threshold is modified), and by how much the threshold changes on average.

     In future could make a variant operation that optimizes a and b to minimize the
     quantity meanqover/pkick (hugged the shape as close as possible with the
     minimum number of kicks), and returns a and b...?
     '''
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
