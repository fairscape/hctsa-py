def wentropy(x,entType = 'shannon',addiontalParameter = None):

    if entType == 'shannon':

        x = np.power(x[ x != 0 ],2)

        return - np.sum(np.multiply(x,np.log(x)))

    elif entType == 'threshold':

        if addiontalParameter is None or isinstance(addiontalParameter,str):

            return None

        x = np.absolute(x)

        return np.sum((x > addiontalParameter))

    elif entType == 'norm':

        if addiontalParameter is None or isinstance(addiontalParameter,str) or addiontalParameter < 1:

            return None

        x = np.absolute(x)

        return np.sum(np.power(x,addiontalParameter))

    elif entType == 'sure':

        if addiontalParameter is None or isinstance(addiontalParameter,str):

            return None

        N = len(x)

        x2 = np.square(x)

        t2 = addiontalParameter**2

        xgt = np.sum((x2 > t2))

        xlt = N - xgt


        return N - (2*xlt) + (t2 *xgt) + np.sum(np.multiply(x2,(x2 <= t2)))

    elif entType == 'logenergy':

        x = np.square(x[x != 0])

        return np.sum(np.log(x))

    else:
        print("invalid entropy type")
        return None
