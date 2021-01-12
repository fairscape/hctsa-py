#© 2020 By The Rector And Visitors Of The University Of Virginia

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

def DN_CompareKSFit(x,whatDist = 'norm'):
    xStep = np.std(x) / 100
    if whatDist == 'norm':
        a, b = stats.norm.fit(x)
        peak = stats.norm.pdf(a,a,b)
        thresh = peak / 100
        xf1 = np.mean(x)
        ange = 10
        while ange > thresh:
            xf1 = xf1 - xStep
            ange = stats.norm.pdf(xf1,a,b)
        ange = 10
        xf2 = np.mean(x)
        while ange > thresh:
            xf2 = xf2 + xStep
            ange = stats.norm.pdf(xf2,a,b)


    #since some outliers real far away can take long time
    #should probably do pre-proccessing before functions
    if whatDist == "uni":

        a,b = stats.uniform.fit(x)
        peak = stats.uniform.pdf(np.mean(x),a,b-a)
        thresh = peak / 100
        xf1 = np.mean(x)
        ange = 10
        while ange > thresh:
            xf1 = xf1 - xStep
            ange = stats.norm.pdf(xf1,a,b)
        ange = 10
        xf2 = np.mean(x)
        while ange > thresh:
            xf2 = xf2 + xStep
            ange = stats.norm.pdf(xf2,a,b)

    #might over write y since changing x
    if whatDist == 'beta':
        scaledx = (x - np.min(x) + .01*np.std(x)) / (np.max(x)-np.min(x)+.02*np.std(x))
        xStep = np.std(scaledx) /100
        a = stats.beta.fit(scaledx)
        b = a[2]
        a = a[1]
        thresh = 1E-5
        xf1 = np.mean(scaledx)
        ange = 10
        while ange > thresh:
            xf1 = xf1 - xStep
            ange = stats.beta.pdf(xf1,a,b)
        ange = 10
        xf2 = np.mean(scaledx)
        while ange > thresh:
            xf2 = xf2 + xStep
            ange = stats.beta.pdf(xf2,a,b)
        x = scaledx


    kde = stats.gaussian_kde(x)
    test_space = np.linspace(np.min(x),np.max(x),1000)
    kde_est = kde(test_space)
    if whatDist == 'norm':
        ffit = stats.norm.pdf(test_space,a,b)
    if whatDist == 'uni':
        ffit = stats.uniform.pdf(test_space,a,b-a)
    if whatDist == 'beta':
        ffit = stats.beta.pdf(test_space,a,b)

    out = {}

    out['adiff'] = sum(np.absolute(kde_est - ffit)*(test_space[1]-test_space[0]))

    out['peaksepy'] = np.max(ffit) - np.max(kde_est)

    r = (ffit != 0)

    out['relent'] = sum(np.multiply(kde_est[r],np.log(np.divide(kde_est[r],ffit[r])))*(test_space[1]-test_space[0]))

    return out
