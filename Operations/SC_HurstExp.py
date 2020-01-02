#import matplotlib.pyplot as plt
def SC_HurstExp(x):

    N = len(x)

    splits = int(np.log2(N))

    rescaledRanges = []

    n = []

    for i in range(splits):

        chunks = 2**(i)

        n.append(int(N / chunks))


        y = x[:N - N % chunks]

        y = y.reshape((chunks,int(N/chunks)))

        m = y.mean(axis = 1,keepdims = True)

        y = y - m

        z = np.cumsum(y,1)

        R = np.max(z,1) - np.min(z,1)

        S = np.std(y,1)

        S[S == 0] = 1


        rescaledRanges.append(np.mean(R/S))

    logRS = np.log(rescaledRanges)
    logn = np.log(n)

    # plt.plot(logn,logRS)
    # plt.show()

    p = np.polyfit(logn,logRS,1)

    return p[0]
