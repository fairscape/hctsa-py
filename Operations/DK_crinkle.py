def DK_crinkle(x):

    x = x - np.mean(x)

    x2 = np.mean(np.square(x))**2

    if x2 == 0:
        return 0
        
    d2 = 2*x[1:-1] - x[0:-2] - x[2:]

    return np.mean(np.power(d2,4)) / x2
