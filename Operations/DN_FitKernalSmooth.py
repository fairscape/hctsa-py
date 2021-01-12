#© 2020 By The Rector And Visitors Of The University Of Virginia

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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
