
def CO_AutoCorr(y,lag = 1,method = 'Fourier',t=1):

    # if not isinstance(y,np.ndarray):
    #     y = np.asarray(y)

    if method == 'TimeDomainStat':

        if lag == []:

            acf = [1]

            for i in range(1,len(y)-1):

                acf.append(np.corrcoef(y[:-i],y[i:])[0,1])

            return acf

        return(np.corrcoef(y[:-lag],y[lag:])[0,1])

    else:

        N = len(y)

        nFFT = int(2**(np.ceil(np.log2(N)) + 1))

        F = np.fft.fft(y - y.mean(),nFFT)

        F = np.multiply(F,np.conj(F))

        acf = np.fft.ifft(F)

        if acf[0] == 0:

            if lag == []:

                return acf

            return acf[lag]

        acf = acf / acf[0]
        acf = acf.real

        if lag == []:

            return acf

        if lag > N:
            pass
            #print("Lag larger than series")

            return

        return acf[lag]
