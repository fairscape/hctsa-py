def CO_trev(y,tau = 'ac'):
        if tau == 'ac':
            tau = CO_FirstZero(y,'ac')
        else:
            tau = CO_FirstMin(y,'mi')
        N = len(y)
        yn = y[0:N-tau]
        yn1 = y[tau:N]
        raw = np.mean(np.power(yn1-yn,3)) / np.mean(np.power(yn1 - yn,2))**(3/2)

        return raw
