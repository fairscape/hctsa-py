#© 2020 By The Rector And Visitors Of The University Of Virginia

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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


def run_histogram_algos(y,results = {},impute = False):

    if impute:
        y = impute(y)
    else:
        y = y[~np.isnan(y)]


    results['DN_Mean'] = DN_Mean(y)

    results['DN_Range'] = DN_Range(y)


    results['DN_IQR'] = DN_IQR(y)


    results['DN_Median'] = DN_Median(y)


    results['DN_Max'] = DN_MinMax(y)
    results['DN_Min'] = DN_MinMax(y,'min')

    results['DN_Mode'] = DN_Mode(y)

    results['DN_Cumulants_cumWhatMay_skew1'] = DN_Cumulants(y,'skew1')
    results['DN_Cumulants_cumWhatMay_skew2'] = DN_Cumulants(y,'skew2')
    results['DN_Cumulants_cumWhatMay_kurt1'] = DN_Cumulants(y,'kurt1')
    results['DN_Cumulants_cumWhatMay_kurt2'] = DN_Cumulants(y,'kurt2')


    results['DN_Burstiness'] = DN_Burstiness(y)

    results['DN_Unique'] = DN_Unique(y)

    results['DN_Within_p_1'] = DN_Withinp(y)
    results['DN_Within_p_2'] = DN_Withinp(y,2)


    results['EN_ShannonEn'] = EN_ShannonEn(y)


    results['DN_STD'] = DN_STD(y)
    if results['DN_STD'] == 0:
        return results

    results['DN_Moments_theMom_2'] = DN_Moments(y,2)
    results['DN_Moments_theMom_3'] = DN_Moments(y,3)
    results['DN_Moments_theMom_4'] = DN_Moments(y,4)
    results['DN_Moments_theMom_5'] = DN_Moments(y,5)
    results['DN_Moments_theMom_6'] = DN_Moments(y,6)

    results['DN_pleft'] = DN_pleft(y)

    results['DN_CustomSkewness'] = DN_CustomSkewness(y)

    results['DN_HighLowMu'] = DN_HighLowMu(y)

    results['DN_nlogL_norm'] = DN_nlogL_norm(y)

    results['DN_Quantile_q_50'] = DN_Quantile(y)
    results['DN_Quantile_q_75'] = DN_Quantile(y,.75)
    results['DN_Quantile_q_90'] = DN_Quantile(y,.90)
    results['DN_Quantile_q_95'] = DN_Quantile(y,.95)
    results['DN_Quantile_q_99'] = DN_Quantile(y,.99)


    out = DN_RemovePoints(y,p = .5)
    results = parse_outputs(out,results,'DN_RemovePoints')


    results['DN_Spread_spreadMeasure_mad'] = DN_Spread(y,'mad')
    results['DN_Spread_spreadMeasure_mead'] = DN_Spread(y,'mead')


    results['DN_TrimmedMean_n_50'] = DN_TrimmedMean(y,.5)
    results['DN_TrimmedMean_n_75'] = DN_TrimmedMean(y,.75)
    results['DN_TrimmedMean_n_25'] = DN_TrimmedMean(y,.25)

    results['DN_cv_k_1'] = DN_cv(y)
    results['DN_cv_k_2'] = DN_cv(y,2)
    results['DN_cv_k_3'] = DN_cv(y,3)

    return results

def time_series_dependent_algos(y,results):
    if np.count_nonzero(np.isnan(y)) > 0:
        #print(y)
        #print(t)
        raise Exception('Missing Value')
    #print('Corr')

    corr = CO_AutoCorr(y,[],'Forier')

    i = 0

    for c in corr:
        if i > 25:
            break
        elif i == 0:
            i = i + 1
            continue

        results['CO_AutoCorr_lag_ ' + str(i)] = c
        i = i + 1


    results['CO_f1ecac'] = CO_f1ecac(y)
    results['CO_FirstMin'] = CO_FirstMin(y)
    results['CO_FirstZero'] = CO_FirstZero(y)

    for alpha in range(1,5):
        for beta in range(1,5):
            results['CO_glscf ' + str(alpha) + ' ' + str(beta)] = CO_glscf(y,alpha,beta)

    results['CO_tc3'] = CO_tc3(y)
    results['CO_trev'] = CO_trev(y)


    out = DN_CompareKSFit(y)
    results = parse_outputs(out,results,'DN_CompareKSFit')

    results['DT_IsSeasonal?'] = DT_IsSeasonal(y)

    results['EN_ApEn'] = EN_ApEn(y)

    out = EN_CID(y)
    results = parse_outputs(out,results,'EN_CID')

    results['EN_PermEn 2, 1'] = EN_PermEn(y)
    results['EN_PermEn 3, 6'] = EN_PermEn(y,3,6)

    out = EN_SampEn(y)
    results['EN_SampEN Sample Entropy'] = out["Sample Entropy"]
    results["EN_SampEN Quadratic Entropy"] = out["Quadratic Entropy"]


    out = IN_AutoMutualInfo(y)
    results = parse_outputs(out,results,'IN_AutoMutualInfo Auto Mutual Info')

    if not BF_iszscored(y):
        out = SY_Trend((y-np.mean(y)) / np.std(y))
    else:
        out = SY_Trend(y)
    results = parse_outputs(out,results,'SY_Trend')

    return results

def round2(y,results = {}):
    #sresults = {}
    if np.count_nonzero(np.isnan(y)) > 0:
        y = impute(y,np.nan)
    start = time.time()
    out  = FC_Suprise(y)
    results = parse_outputs(out,results,'FC_Suprise')
    results['FC_Suprise Time'] = time.time() - start
    start = time.time()
    for i in range(2,4):
        for j in range(2,6):
            out = EN_PermEn(y,i,j)
            if isinstance(out,dict):
                results = parse_outputs(out,results,'EN_PermEn '+ str(i) + ' ,'  + str(j))
    results['EN_PermEm Time'] = time.time() - start
    start = time.time()
    for i in range(3,5):
        for j in [.15,.3]:
            out = EN_SampEn(y,i,j)
            results['EN_Samp Sample Entropy ' + str(i) + ' ' + str(j)] = out["Sample Entropy"]
            results["EN_Samp Quadratic Entropy "+ str(i) + ' ' + str(j)] = out["Quadratic Entropy"]
    results['EN_SampEn Time'] = time.time() - start
    start = time.time()
    try:
        out = MD_hrv_classic(y)
        results = parse_outputs(out,results,'MD_hrv')
    except:
        print('Failed hrv')
    out = MD_pNN(y)
    results = parse_outputs(out,results,'MD_pNN')

    results['SC_HurstExp'] = SC_HurstExp(y)
    results['Med Time'] = time.time() - start
    # results['SC_DFA'] = SC_DFA(y)
    start = time.time()
    for i in range(2,4):
        for j in [.2]:

            out = EN_mse(y,range(2,8),i,j)

            results = parse_outputs(out,results,'EN_mse '+ str(i) + ' ,'  + str(j))
    results['EN_mse Time'] = time.time() - start

    start = time.time()
    for n in [10,20,50,100,250]:
        out = SY_LocalGlobal(y,'l',n)
        if isinstance(out,dict):
            results = parse_outputs(out,results,'SY_LocalGlobal_l' + str(n))

    for n in [.05,.1,.2,.5]:
        out = SY_LocalGlobal(y,'p',n)
        if isinstance(out,dict):
            results = parse_outputs(out,results,'SY_LocalGlobal_p' + str(n))

    for n in [10,20,50,100,250]:
        out = SY_LocalGlobal(y,'unicg',n)
        if isinstance(out,dict):
            results = parse_outputs(out,results,'SY_LocalGlobal_unicg' + str(n))

    for n in [10,20,50,100,250]:
        out = SY_LocalGlobal(y,'randcg',n)
        if isinstance(out,dict):
            results = parse_outputs(out,results,'SY_LocalGlobal_randcg' + str(n))
    results['SY_LocalGlobal Time'] = time.time() - start
    start = time.time()
    for i in range(0,16):
        try:
            results['CO_RM_AMInformation ' + str(i)] = CO_RM_AMInformation(y,i)
        except:
            results['CO_RM_AMInformation ' + str(i)] = np.nan
            continue
    results['CO_RM_AMInformation Time']= time.time() - start
    start = time.time()
    for i in range(2,6):

        for tau in range(1,5):

            out = SB_TransitionMatrix(y,'quantile',i,tau)
            results = parse_outputs(out,results,'SB_TransitionMatrix' + str(i) + str(tau))
    results['SB_TransitionMatrix Time'] = time.time() - start

    start = time.time()
    for i in [25,50,100,150,200]:

        out = SY_SpreadRandomLocal(y,i)

        if isinstance(out,dict):
            results = parse_outputs(out,results,'SY_SpreadRandomLocal' + str(i))

    results['SY_SpreadRandomLocal Time'] = time.time() - start
    start = time.time()
    for l in ['l','n']:

        for n in [25,50,75,100]:

            out = ST_LocalExtrema(y,l,n)
            if isinstance(out,dict):
                results = parse_outputs(out,results,'ST_LocalExtrema_' + l + str(n))
    results['ST_LocalExtrema Time'] = time.time() - start

    start = time.time()
    for prop in ['biasprop','momentum','runningvar','prop']:
        if prop == 'prop':
            parameters = [[.1],[.5],[.9]]
        elif prop == 'biasprop':
            parameters = [[.5,.1],[.1,.5]]
        elif prop == 'momentum':
            parameters = [[2],[5],[10]]
        elif prop == 'runningvar':
            parameters = [[]]
        for para in parameters:
            out = PH_Walker(y,prop,para)
            if isinstance(out,dict):
                results = parse_outputs(out,results,'PH_Walker' + str(prop) + str(para))
    results['PH_Walker Time'] = time.time() - start
    out = SY_PeriodVital(y)
    results = parse_outputs(out,results,'SY_PeriodVital')

    return results


def round3(y,results = {}):

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
        results['DN_Observations'] = DN_ObsCount(y)
        return results

    results['DN_Observations'] = DN_ObsCount(y)

    results = run_histogram_algos(y,results)
    if results['DN_STD'] != 0.0:
        y_imputed = impute(y,last_non_nan)
        results = time_series_dependent_algos(impute(y,last_non_nan),results)
        results = round2(y_imputed,results)
    results = round3(impute(y,last_non_nan),results)

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
