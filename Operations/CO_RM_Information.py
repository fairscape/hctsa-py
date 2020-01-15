def CO_RM_AMInformation(y,tau = 1):

    if tau >= len(y):
        return None

    y1 = y[:-tau]
    y2 = y[tau:]

    out = RM_information(y1,y2)

    return out
