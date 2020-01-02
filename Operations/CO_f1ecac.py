import math
def CO_f1ecac(y):
    N = len(y)
    thresh = 1 / math.exp(1)
    for i in range(1,N):
        auto = CO_AutoCorr(y,i)
        if ( auto - thresh ) < 0:
            return i
    return N
