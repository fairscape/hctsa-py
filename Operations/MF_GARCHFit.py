import arch
from pprint import pprint

from statsmodels.stats.diagnostic import het_arch

def MF_GARCHFit(y,preproc = None,P = 1,Q = 1):

    y = (y - np.mean(y)) / np.std(y)

    N = len(y)

    outDict = {}

    lm, lmpval,fval,fpval = het_arch(y)

    outDict['lm'] = lm
    outDict['lmpval'] = lmpval
    outDict['fval'] = fval
    outDict['fpval'] = fpval

    model= arch.arch_model(y, vol='Garch', p=P, o=0, q=Q, dist='Normal')
    results=model.fit()

    #print(results.summary())

    params = results._params
    paraNames = results._names

    outDict['logl'] = results._loglikelihood
    outDict['success'] = results._optim_output['success']

    for i in range(len(params)):

        outDict[paraNames[i]] = params[i]

    #pprint(vars(results))

    return outDict
