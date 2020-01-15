def MD_polvar(x,d = 1 ,D = 6):

    dx = np.absolute(np.diff(x))

    N = len(dx)

    xsym = ( dx >= d )
    zseq = np.zeros(D)
    oseq = np.ones(D)

    i = 1
    pc = 0

    while i <= (N-D):

        xseq = xsym[i:(i+D)]

        if np.sum((xseq == zseq)) == D or np.sum((xseq == oseq)) == D:

            pc = pc + 1
            i = i + D

        else:

            i = i + 1

    p = pc / N

    return p
