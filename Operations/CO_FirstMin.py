def CO_FirstMin(y, minWhat = 'ac'):
    if minWhat == 'mi':
        x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
        acf = IN_AutoMutualInfo(y,x,'gaussian')
    else:
        acf = CO_AutoCorr(y,[],'Fourier')
    N = len(y)

    for i in range(1,N-1):
        if i == 2 and (acf[2] > acf[1]):
            return 1
        elif (i > 2) and (acf[i-2] > acf[i-1]) and (acf[i-1] < acf[i]):
            return i-1
    return N
