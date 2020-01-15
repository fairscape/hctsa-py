def DK_lagembed(x,M,lag = 1):

    advance = 0

    #Should make sure x is column
    #for me pretty much always just array doesnt matter

    lx = len(x)

    newsize = lx - lag*(M-1)

    y = np.zeros((newsize,M))

    i = 0

    for j in range(0,lag*(-(M)),-lag):

        first =  lag*(M-1) + j

        last = first + newsize

        if last > lx:

            last = lx - 1

        y[:,i] = x[first:last]

        i = i + 1

    return y
