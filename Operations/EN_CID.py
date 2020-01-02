def EN_CID(y):


    CE1 = f_CE1(y)
    CE2 = f_CE2(y)

    minCE1 = f_CE1(np.sort(y))
    minCE2 = f_CE2(np.sort(y))

    CE1_norm = CE1 / minCE1
    CE2_norm = CE2 / minCE2

    out = {'CE1':CE1,'CE2':CE2,'minCE1':minCE1,'minCE2':minCE2,
            'CE1_norm':CE1_norm,'CE2_norm':CE2_norm}
    return out

def f_CE1(y):
    return np.sqrt(np.mean(np.power(np.diff(y),2) ) )

def f_CE2(y):
    return np.mean(np.sqrt( 1 + np.power(np.diff(y),2) ) )
