def CO_RM_AMInformation(y,tau = 1):
    """
    A wrapper for rm_information(), which calculates automutal information

    Inputs:
        y, the input time series
        tau, the time lag at which to calculate automutal information

    :returns estimate of mutual information

    - Wrapper initially developed by Ben D. Fulcher in MATLAB
    - rm_information.py initially developed by Rudy Moddemeijer in MATLAB
    - Translated to python by Tucker Cullen

    """

    if tau >= len(y):

        return 

    y1 = y[:-tau]
    y2 = y[tau:]

    out = RM_information(y1,y2)

    return out[0]
