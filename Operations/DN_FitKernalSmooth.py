from Periphery import *
def DN_FitKernalSmooth(x,varargin = {}):
    #varargin should be dict with possible keys numcross
    #area and arclength

    out = {}

    m = np.mean(x)

    kde = stats.gaussian_kde(x)
    #i think matlabs kde uses 100 points
    #but end numbers end up being midly off
    #seems to be rounding entropy max, min line up
    test_space = np.linspace(np.min(x),np.max(x),100)

    f = kde(test_space)

    df = np.diff(f)

    ddf  = np.diff(df)

    sdsp = ddf[BF_sgnchange(df,1)]

    out['npeaks'] = sum(sdsp < -.0002)

    out['max'] = np.max(f)

    out['entropy'] = - sum(np.multiply(f[f>0],np.log(f[f>0])))*(test_space[2]-test_space[1])

    out1 = sum(f[test_space > m]) * (test_space[2]-test_space[1])
    out2 = sum(f[test_space < m]) * (test_space[2]-test_space[1])
    out['asym'] = out1 / out2

    out1 = sum(np.absolute(np.diff(f[test_space < m]))) * (test_space[2]-test_space[1])
    out1 = sum(np.absolute(np.diff(f[test_space > m]))) * (test_space[2]-test_space[1])
    out['plsym'] = out1 / out2

    if 'numcross' in varargin:
        thresholds = varargin['numcross']
        out['numCrosses']  = {}
        for i in range(len(thresholds)):
            numCrosses = sum(BF_sgnchange(f - thresholds[i]))
            out['numCrosses'][thresholds[i]] = numCrosses
    if 'area' in varargin:
        thresholds = varargin['area']
        out['area']  = {}
        for i in range(len(thresholds)):
            areaHere = sum(f[f < thresholds[i]]) * (test_space[2]-test_space[1])
            out['area'][thresholds[i]] = areaHere
    if 'arclength' in varargin:
        thresholds = varargin['arclength']
        out['arclength']  = {}
        for i in range(len(thresholds)):
            fd = np.absolute(np.diff(f[(test_space > m - thresholds[i]) & (test_space < m + thresholds[i])]))
            arclengthHere = sum(fd) * (test_space[2]-test_space[1])
            out['arclength'][thresholds[i]] = arclengthHere
    return out
