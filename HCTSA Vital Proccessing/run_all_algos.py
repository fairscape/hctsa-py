import numpy as np
import multiprocessing as mp
from functools import partial
import pandas as pd
import csv
#import make_operations
#operations = make_operations.make_operations()
#make_operations.make_otherfunctions()
from Operations import *
from Periphery import *
import time

def round3(y):

    results = {}

    taus = [1,2,3,4,5,'tau']

    for tau in taus:

        out = CO_Embed2_Basic(y,tau)

        if isinstance(out,dict):

            results = parse_outputs(out,results,'CO_Embed2_Basic_tau' + str(tau))

    how = ['absfar','absclose','min','max']
    ps = [.2,.6,.8]

    for h in how:

        for p in ps:

            out = DN_RemovePoints(y,h,p)

            if isinstance(out,dict):

                results = parse_outputs(out,results,'DN_RemovePoints' + str(h) + '_' + str(p))


    results['DK_crinkle'] = DK_crinkle(y)


    results['DK_theilerQ'] = DK_theilerQ(y)


    binarizeHows = ['diff','mean','median','iqr']

    for binary in binarizeHows:

        out = SB_BinaryStats(y,binary)

        if isinstance(out,dict):

            results = parse_outputs(out,results,'SB_BinaryMethod_' + str(binary))


    types = ['seg','len']
    numSegs = [2,3,4,5,10]

    for ty in types:

        for seg in numSegs:

            out = SY_StatAv(y,ty,seg)

            if isinstance(out,dict):

                results = parse_outputs(out,results,'SY_StatAv' + str(ty) + '_ ' + str(seg) + '_')

    samplAs = [.25,.5,1,2,3]
    bs = [.01,.05,.1,.25]

    for a in samplAs:

        for b in bs:

            out = EX_MovingThreshold(y,a,b)

            if isinstance(out,dict):

                results = parse_outputs(out,results,'EX_MovingThreshold_a' + str(a)+ '_b' + str(b))


    binaryMethods = ['diff','mean','median','iqr']

    for bMethod in binaryMethods:



        out = SB_MotifTwo(y,bMethod)

        if isinstance(out,dict):

            results = parse_outputs(out,results,'SB_MotifTwo_'  + bMethod)

    binaryMethods = ['diffquant','quantile']

    for bMethod in binaryMethods:

        out = SB_MotifThree(y,bMethod)

        if isinstance(out,dict):

            results = parse_outputs(out,results,'SB_MotifThree_'  + bMethod )

    out = SY_RangeEvolve(y)

    results = parse_outputs(out,results,'SY_RangeEvolve')

    for i in range(1,26):

        results['SY_StdNthDer_' + str(i)] = SY_StdNthDer(y,i)

    taus = [[1,2,3],[2,3,4,5],[1,5,10]]

    for tau in taus:


        results['CO_NonlinearAuto_' + str(tau).replace(' ','').replace(',','_')] = CO_NonlinearAutocorr(y,tau)


    for tau in [1,2,5,25,50]:

        out = CO_tc3(y,tau)

        if isinstance(out,dict):

            results = parse_outputs(out,results,'CO_tc3_' + str(tau) + '_')

        out = CO_trev(y,tau)

        if isinstance(out,dict):

            results = parse_outputs(out,results,'CO_trev_'  + str(tau) + '_')

    entropies = ['shannon','logenergy','threshold','sure','norm']

    for entropy in entropies:

        results['EN_wentropy_' + entropy] = EN_wentropy(y,entropy)


    out = SY_DriftingMean(y)

    if isinstance(out,dict):

        results = parse_outputs(out,results,'SY_DriftingMean_')


    return results


def run_algos(y,algos = 'all',last_non_nan = np.nan,t=1):

    results = {}

    if np.count_nonzero(~np.isnan(y)) < 10:

        return {}

    results = round3(impute(y,last_non_nan))

    results['time'] = t

    return results

def parse_outputs(outputs,results,func):
    for key in outputs:
        if isinstance(outputs[key],list) or isinstance(outputs[key],np.ndarray):
            i = 1
            for out in outputs[key]:
                results[func + ' ' + key + ' ' + str(i)] = out
                i = i + 1
        else:
            results[func + ' ' + key] = outputs[key]
    return results




def get_interval(interval_length,end_time,times):
    # interval_length is in seconds
    # end_time is in seconds
    return np.where((times <= end_time) & (times > end_time - interval_length))


def all_times(t,series,time,interval_length = 600,algos = 'all'):
    indx = get_interval(interval_length,int(t),time)
    indx = indx[0]
    if len(indx) <= 1:
        return {'time':t}
    if np.isnan(series[np.min(indx)]):
        nonnan = np.argwhere(~np.isnan(series))[np.argwhere(~np.isnan(series)) < np.min(indx)]
        if len(nonnan) != 0:
            last_non_nan_indx = np.max(nonnan)
            lastvalue = series[last_non_nan_indx]
        else:
            lastvalue = np.nan
        results = run_algos(series[indx],algos,lastvalue,t)
        #results = run_algos(series[indx],['DN_ObsCount'],lastvalue,t)
    else:
        #results = run_algos(series[indx],['DN_ObsCount'],1,t)
        results = run_algos(series[indx],algos,1,t)
    results['time'] = t
    return results


def impute(y_test,last):
    if np.isnan(y_test[0]) and np.isnan(last):

        min = np.min(np.argwhere(~np.isnan(y_test)))
        return y_test[min:]
    elif np.isnan(y_test[0]):
        y_test[0] = last
    y_test = nanfill(y_test)
    return y_test

def nanfill(x):
    for i in np.argwhere(np.isnan(x)):
        x[i] = x[i-1]
    return x

def run_all(time_series,time1,id,interval_length = 60*10,step_size=60*10,algos = 'all'):
    end_times = np.arange(np.min(time1) + interval_length,np.max(time1),step_size)
    if not isinstance(time_series,dict):
        time_series = {'y':time_series}
    full_results = {}
    for key, data in time_series.items():
        print("Analyzing " + key)

        np.seterr(divide='ignore')
        start = time.time()
        pool = mp.Pool(mp.cpu_count())
        #results = [pool.apply(all, args=(hr,time,interval_length,t)) for t in end_times]
        results = pool.map(partial(all_times,series = data,time = time1,algos = algos), [t for t in end_times])
        pool.close()

        print("Performing Calcs took " + str(time.time() - start))
        #results = all_times(end_times[1],data,time)

        max = 0
        start = time.time()
        for guy in results:
            if len(guy) > max:
                columns = list(guy.keys())
                max = len(guy)
        #print('Loop through results took ' + str(time.time() - start))
        #print('Number of outputs is: ' + str(len(columns)))
        for result in results:
            for column in columns:
                if column not in result.keys():
                    result[column] = ''

        with open('./Results/UVA_' + str(id) + '_' + key + '3.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)

    return full_results
