def DK_timerev(x,timeLag = 1):

    foo = DK_lagembed(x,3,timeLag)

    a = foo[:,0]
    b = foo[:,1]
    c = foo[:,2]

    res = np.mean(np.multiply(np.multiply(a,a),b) - np.multiply(np.multiply(b,c),c))

    return res
