def MF_ARMA_orders(y,pr = [1,2,3,4,5],qr=[0,1,2,3,4,5]):

    y = (y - np.mean(y)) / np.std(y)

    aics = np.zeros((len(pr),len(qr)))
    bics = np.zeros((len(pr),len(qr)))

    for i in range(len(pr)):

        for j in range(len(qr)):

            p = pr[i]
            q = qr[i]

            try:

                model = ARIMA(y, order=(p,0,q))
                model_fit = model.fit( disp=False)

            except:
                print("FAILED ARMA MODEL")
                return None
            aics[i,j] = model_fit.aic
            bics[i,j] = model_fit.bic

    outDict = {}

    outDict['aic_min'] = np.min(aics)

    mins = np.argwhere(aics == np.min(aics))[0]

    outDict['opt_p'] = pr[mins[0]]

    outDict['opt_q'] = qr[mins[0]]

    outDict['meanAICS'] = np.mean(aics)
    outDict['stdAICS'] = np.std(aics)

    outDict['meanBICS'] = np.mean(bics)
    outDict['stdBICS'] = np.std(bics)

    return outDict
