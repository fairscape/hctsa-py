def SB_BinaryStats(y,binaryMethod = 'diff'):

    yBin = BF_Binarize(y,binaryMethod)

    N = len(yBin)

    outDict = {}

    outDict['pupstat2'] = np.sum((yBin[math.floor(N /2):] == 1))  / np.sum((yBin[:math.floor(N /2)] == 1))

    stretch1 = []
    stretch0 = []
    count = 1



    for i in range(1,N):

        if yBin[i] == yBin[i - 1]:

            count = count + 1

        else:

            if yBin[i - 1] == 1:

                stretch1.append(count)

            else:

                stretch0.append(count)
            count = 1
    if yBin[N-1] == 1:

        stretch1.append(count)

    else:

        stretch0.append(count)


    outDict['pstretch1'] = len(stretch1) / N

    if stretch0 == []:

        outDict['longstretch0'] = 0
        outDict['meanstretch0'] = 0
        outDict['stdstretch0'] = None

    else:

        outDict['longstretch0'] = np.max(stretch0)
        outDict['meanstretch0'] = np.mean(stretch0)
        outDict['stdstretch0'] = np.std(stretch0,ddof = 1)

    if stretch1 == []:

        outDict['longstretch1'] = 0
        outDict['meanstretch1'] = 0
        outDict['stdstretch1'] = None

    else:

        outDict['longstretch1'] = np.max(stretch1)
        outDict['meanstretch1'] = np.mean(stretch1)
        outDict['stdstretch1'] = np.std(stretch1,ddof = 1)

    try:

        outDict['meanstretchdiff'] = outDict['meanstretch1'] - outDict['meanstretch0']
        outDict['stdstretchdiff'] = outDict['stdstretch1'] - outDict['stdstretch0']

    except:

        pass


    return outDict
