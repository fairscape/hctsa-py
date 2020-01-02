from scipy import optimize
def DT_IsSeasonal(y):

    N = len(y)

    th_fit = 0.3
    th_ampl = 0.5

    try:
        params, params_covariance = optimize.curve_fit(test_func, np.arange(N), y, p0=[10, 13,600,0])
    except:
        return False

    a,b,c,d = params



    y_pred = a * np.sin(b * np.arange(N) + d) + c

    SST = sum(np.power(y - np.mean(y),2))
    SSr = sum(np.power(y - y_pred,2))

    R = 1 - SSr / SST


    if R > th_fit: #and (np.absolute(a) > th_ampl*.1*np.std(y)):
        return True
    else:
        return False

def test_func(x, a, b,c,d):
    return a * np.sin(b * x + d) + c
