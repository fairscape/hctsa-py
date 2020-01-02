from scipy import signal
def SY_Trend(y):

    N  = len(y)
    stdRatio = np.std(signal.detrend(y)) / np.std(y)

    gradient, intercept = LinearFit(np.arange(N),y)

    yC = np.cumsum(y)
    meanYC = np.mean(yC)
    stdYC = np.std(yC)

    #print(gradient)
    #print(intercept)

    gradientYC, interceptYC = LinearFit(np.arange(N),yC)

    meanYC12 = np.mean(yC[0:int(np.floor(N/2))])
    meanYC22 = np.mean(yC[int(np.floor(N/2)):])

    out = {'stdRatio':stdRatio,'gradient':gradient,'intercept':intercept,
            'meanYC':meanYC,'stdYC':stdYC,'gradientYC':gradientYC,
            'interceptYC':interceptYC,'meanYC12':meanYC12,'meanYC22':meanYC22}
    return out

def LinearFit(xData,yData):
    m, b = np.polyfit(xData,yData,1)
    return m,b
