from statsmodels.tsa.arima_model import ARIMA
def MF_AR_arcov(y,p = 2):


    try:

        model = ARIMA(y, order=(p,0,0))
        model_fit = model.fit( disp=False)

    except:
        #Non-stationary returns expception
        return

    ar_coefs = model_fit.arparams
    coef_errors = model_fit.bse

    outDict = {}

    stable = True

    for num in model_fit.arroots:

        if np.absolute(num) < 1:

            stable = False

    outDict['stable'] = stable

    y_est = model_fit.fittedvalues[-1]

    err = y - y_est

    for i in range(len(ar_coefs)):

        outDict['ar' + str(i  + 1)] = ar_coefs[i]
        outDict['ar error' + str(i + 1)] = coef_errors[i]

    outDict['res_mu'] = np.mean(err)
    outDict['res_std'] = np.std(err)

    outDict['res_AC1'] = CO_AutoCorr(err,1)
    outDict['res_AC2'] = CO_AutoCorr(err,2)

    return outDict
