def SB_MotifTwo(y,binarizeHow = 'diff'):

    yBin = BF_Binarize(y,binarizeHow)

    N = len(yBin)

    r1 = (yBin == 1)

    r0 = (yBin == 0)

    outDict = {}

    outDict['u'] = np.mean(r1)
    outDict['d'] = np.mean(r0)

    pp  = np.asarray([ np.mean(r1), np.mean(r0)])

    outDict['h'] = f_entropy(pp)

    r1 = r1[:-1]
    r0 = r0[:-1]

    rs1 = [r0,r1]

    rs2 = [[0,0],[0,0]]
    pp = np.zeros((2,2))

    letters = ['d','u']

    for i in range(2):

        l1 = letters[i]

        for j in range(2):

            l2 = letters[j]

            rs2[i][j] = np.logical_and(rs1[i],yBin[1:] == j)

            outDict[l1 + l2] = np.mean(rs2[i][j])

            pp[i,j] = np.mean(rs2[i][j])

            rs2[i][j] = rs2[i][j][:-1]

    outDict['hh'] = f_entropy(pp)

    rs3 = [[[0,0],[0,0]],[[0,0],[0,0]]]
    pp = np.zeros((2,2,2))

    for i in range(2):

        l1 = letters[i]

        for j in range(2):

            l2 = letters[j]

            for k in range(2):

                l3 = letters[k]

                rs3[i][j][k] =np.logical_and(rs2[i][j],yBin[2:] == k)

                outDict[l1 + l2 + l3] = np.mean(rs3[i][j][k])

                pp[i,j,k] = np.mean(rs3[i][j][k])

                rs3[i][j][k] = rs3[i][j][k][:-1]

    outDict['hhh'] = f_entropy(pp)

    rs4 = [[[[0,0],[0,0]],[[0,0],[0,0]]],[[[0,0],[0,0]],[[0,0],[0,0]]]]
    pp = np.zeros((2,2,2,2))

    for i in range(2):

        l1 = letters[i]

        for j in range(2):

            l2 = letters[j]

            for k in range(2):

                l3 = letters[k]

                for l in range(2):

                    l4 = letters[l]

                    rs4[i][j][k][l] =np.logical_and(rs3[i][j][k],yBin[3:] == l)

                    outDict[l1 + l2 + l3 + l4] = np.mean(rs4[i][j][k][l])

                    pp[i,j,k,l] = np.mean(rs4[i][j][k][l])


    outDict['hhhh'] = f_entropy(pp)


    return outDict
