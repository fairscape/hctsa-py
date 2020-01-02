def EN_mse(y,scale=range(2,11),m=2,r=.15,adjust_r=True):

    minTSLength = 20
    numscales = len(scale)
    y_cg = []

    for i in range(numscales):

        bufferSize = scale[i]
        y_buffer = BF_makeBuffer(y,bufferSize)
        y_cg.append(np.mean(y_buffer,1))

    outEns = []



    for si in range(numscales):
        if len(y_cg[si]) >= minTSLength:

            sampEnStruct = EN_SampEn(y_cg[si],m,r)
            outEns.append(sampEnStruct)
        else:
            outEns.append(np.nan)
    sampEns = []
    for out in outEns:
        if not isinstance(out,dict):
            sampEns.append(np.nan)
            continue
        sampEns.append(out['Sample Entropy'])

    maxSampen = np.max(sampEns)
    maxIndx = np.argmax(sampEns)

    minSampen = np.min(sampEns)
    minIndx = np.argmin(sampEns)

    meanSampen = np.mean(sampEns)

    stdSampen = np.std(sampEns)

    meanchSampen = np.mean(np.diff(sampEns))

    out = {'max Samp En':maxSampen,'max point':scale[maxIndx],'min Samp En':minSampen,\
    'min point':scale[minIndx],'mean Samp En':meanSampen,'std Samp En':stdSampen, 'Mean Change':meanchSampen}

    i = 1
    for sampEn in sampEns:
        out['sampEn ' + str(i)] = sampEn
        i = i + 1

    return out
