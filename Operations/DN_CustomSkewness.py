#© 2020 By The Rector And Visitors Of The University Of Virginia

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#@numba.jit(nopython=True,parallel=True)
def DN_CustomSkewness(y,whatSkew = 'pearson'):
    if whatSkew == 'pearson':
        if np.std(y) != 0:
            return (3*np.mean(y) - np.median(y)) / np.std(y)
        else:
            return 0
    elif whatSkew == 'bowley':
        qs = np.quantile(y,[.25,.5,.75])
        if np.std(y) != 0:
            return (qs[2] + qs[0] - 2*qs[1]) / (qs[2] - qs[0])
        else:
            return 0

    else:
         raise Exception('whatSkew must be either pearson or bowley.')
