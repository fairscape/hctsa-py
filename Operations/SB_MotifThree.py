def SB_MotifThree( y, cgHow = 'quantile'):

    numLetters = 3

    if cgHow == 'quantile':

        yt = SB_CoarseGrain(y,'quantile',numLetters)

    elif cgHow == 'diffquant':

        yt = SB_CoarseGrain(np.diff(y),'quantile',numLetters)

    else:

        return


    N = len(yt)


    r1 = [[],[],[]]

    out1 = np.zeros(3)

    for i in range(1,4):

        r1[i - 1] = np.argwhere(yt == i)[:,0]

        out1[i - 1] = len(r1[i - 1]) / N

    outDict = {}

    outDict['a'] = out1[0]
    outDict['b'] = out1[1]
    outDict['c'] = out1[2]
    outDict['h'] = f_entropy(out1)

    for i in range(3):

        if len(r1[i]) == 0:

            continue

        if r1[i][-1] == N - 1:

            r1[i] = r1[i][:-1]

    r2 =[[[],[],[]],[[],[],[]],[[],[],[]]]



    out2 = np.zeros((3,3))

    for i in range(1,4):

        iIndex = i - 1

        for j in range(1,4):

            jIndex = j - 1

            r2[iIndex][jIndex] = r1[iIndex][np.argwhere(yt[ r1[iIndex] + 1 ] == j)][:,0]

            out2[iIndex,jIndex] = len(r2[iIndex][jIndex]) / (N-1)

    for i in range(3):

        for j in range(3):

            if len(r2[i][j]) == 0:

                continue

            if r2[i][j][-1] == N - 2:

                r2[i][j] = r2[i][j][:-1]





    r3 =[[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]]

    out3 = np.zeros((3,3,3))

    for i in range(1,4):

        iIndex = i - 1

        for j in range(1,4):

            jIndex = j - 1

            for k in range(1,4):

                kIndex = k -1

                r3[iIndex][jIndex][kIndex] = r2[iIndex][jIndex][np.argwhere(yt[ r2[iIndex][jIndex] + 2 ] == k)][:,0]

                out3[iIndex,jIndex,kIndex] = len(r3[iIndex][jIndex][kIndex]) / (N-2)


    letters = ['a','b','c']


    for i in range(3):

        l1 = letters[i]

        for j in range(3):

            l2 = letters[j]

            outDict[l1 + l2] = out2[i,j]

    outDict['hh'] = f_entropy(out2)

    for i in range(3):

        l1 = letters[i]

        for j in range(3):

            l2 = letters[j]

            for k in range(3):

                l3 = letters[k]

                outDict[l1 + l2 + l3] = out3[i,j,k]

    outDict['hhh'] = f_entropy(out3)

    #Seems very ineffiecnt probs can use other numpy functions to speed up
    for i in range(3):

        for j in range(3):

            for k in range(3):

                if len(r3[i][j][k]) == 0:

                    continue


                if r3[i][j][k][-1] == N - 3:

                    r3[i][j][k] = r3[i][j][k][:-1]

    r4 =[[[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]],[[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]],[[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]]]

    out4 = np.zeros((3,3,3,3))

    for i in range(1,4):

        iIndex = i - 1

        for j in range(1,4):

            jIndex = j - 1

            for k in range(1,4):

                kIndex = k -1

                for l in range(1,4):

                    lIndex = l - 1

                    r4[iIndex][jIndex][kIndex][lIndex] = r3[iIndex][jIndex][kIndex][np.argwhere(yt[ r3[iIndex][jIndex][kIndex] + 3 ] == l)][:,0]

                    out4[iIndex,jIndex,kIndex,lIndex] = len(r4[iIndex][jIndex][kIndex][lIndex]) / (N-3)
    for i in range(3):

        l1 = letters[i]

        for j in range(3):

            l2 = letters[j]

            for k in range(3):

                l3 = letters[k]

                for l in range(3):

                    l4 = letters[l]

                    outDict[l1 + l2 + l3 + l4] = out4[i,j,k,l]

    outDict['hhhh'] = f_entropy(out4)


    return outDict


def f_entropy(x):

    h = -np.sum(np.multiply(x[ x > 0],np.log(x[x > 0])))

    return h
