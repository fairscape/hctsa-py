from matplotlib import mlab
def SY_PeriodVital(x):

    f1 = 1
    f2 = 6

    z = np.diff(x)

    [F, t, p] =  signal.spectrogram(z,fs = 60)

    f = np.logical_and(F >= f1,F <= f2)

    p = p[f]

    F = F[f]

    Pmean = np.mean(p)

    Pmax = np.max(p)
    ff = np.argmax(p)
    if ff >= len(F):
        Pf = np.nan
    else:
        Pf = F[ff]
    Pr = Pmax / Pmean
    Pstat = np.log(Pr)

    return {'Pstat':Pstat,'Pmax':Pmax,'Pmean':Pmean,'Pf':Pf}
