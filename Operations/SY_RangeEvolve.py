def SY_RangeEvolve(y):

    N = len(y)

    cums = np.zeros(N)

    for i in range(1,N+1):

        cums[i-1] = np.max(y[:i]) - np.min(y[:i])

    fullr = cums[N-1]

    outDict = {}

    outDict['totnuq'] = lunique(cums)

    outDict['nuqp1'] = cumtox(cums,.01,N,outDict['totnuq'])
    outDict['nuqp10'] = cumtox(cums,.1,N,outDict['totnuq'])
    outDict['nuqp20'] = cumtox(cums,.2,N,outDict['totnuq'])
    outDict['nuqp50'] = cumtox(cums,.5,N,outDict['totnuq'])

    Ns = [10,50,100,1000]

    for n in Ns:

        if N <= n:

            outDict['nuql' + str(n)] = np.nan


        else:

            outDict['nuql' + str(n)] = lunique(cums[:n]) / outDict['totnuq']


    Ns = [10,50,100,1000]

    for n in Ns:

        if N >= n:

            outDict['l' + str(n)] = cums[n - 1] / fullr


        else:

            outDict['l' + str(n)]= np.nan



    outDict['p1'] = cums[math.ceil(N*.01)-1] / fullr
    outDict['p10'] = cums[math.ceil(N*.1)-1] / fullr
    outDict['p20'] = cums[math.ceil(N*.2)-1] / fullr
    outDict['p50'] = cums[math.ceil(N*.5)-1] / fullr



    return outDict

def lunique(x):
    return len(np.unique(x))
def cumtox(cums,x,N,tot):
    return lunique(cums[:math.floor(N*x)]) / tot
