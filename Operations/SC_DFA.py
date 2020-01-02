def SC_DFA(y):

    N = len(y)

    tau = int(np.floor(N/2))

    y = y - np.mean(y)

    x = np.cumsum(y)

    taus = np.arange(5,tau+1)

    ntau = len(taus)

    F = np.zeros(ntau)

    for i in range(ntau):

        t = int(taus[i])



        x_buff = x[:N - N % t]

        x_buff = x_buff.reshape((int(N / t),t))


        y_buff = np.zeros((int(N / t),t))

        for j in range(int(N / t)):

            tt = range(0,int(t))

            p = np.polyfit(tt,x_buff[j,:],1)

            y_buff[j,:] =  np.power(x_buff[j,:] - np.polyval(p,tt),2)



        y_buff.reshape((N - N % t,1))

        F[i] = np.sqrt(np.mean(y_buff))

    logtaur = np.log(taus)

    logF = np.log(F)

    p = np.polyfit(logtaur,logF,1)

    return p[0]
