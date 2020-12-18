import h5py
import numpy as np
#import multiprocessing as mp
import pandas as pd
#import make_operations
#operations = make_operations.make_operations()
#make_operations.make_otherfunctions()
from Operations import *
from Periphery import *
import run_all_algos as run
import sys
import scipy.io

#Reads in houlter data
def read_in_data(id):
    id = str(id).zfill(4)
    data = np.genfromtxt('/Users/justinniestroy-admin/Documents/Work/Randall Data/houlter data/HR/UVA' + id +'_hr.csv', delimiter=',')
    time = data[:,0]
    hr = data[:,1]
    return(time,hr)


#Reads in new nicu data from Doug
#this was for UVA_****_vitals.mat
def read_in_NICU_file(path):
    arrays = {}
    f = h5py.File(path,'r')
    for k, v in f.items():
        if k != 'vdata' and k != 'vt':
            continue
        arrays[k] = np.array(v)
    df = pd.DataFrame(np.transpose(arrays['vdata']))
    df = df.dropna(axis=1, how='all')
    df.columns = get_column_names(f['vname'],f)
    times = pd.Series(arrays['vt'][0], index=df.index)

    return df,times

# Data on Dataverse can be read in using below
#Below is for NICU_****_vitals.mat
def read_NICU_vitals(path):
    f = scipy.io.loadmat(path)
    df = pd.DataFrame(f['vdata'])
    names = []
    for i in range(len(f['vname'])):
        names.append(f['vname'][i][0][0])
    df.columns = names
    times = pd.DataFrame(f['vt'])[0]
    return df, times

def get_column_names(vname,f):
    names = []
    for name in vname[0]:
        obj = f[name]
        col_name = ''.join(chr(i) for i in obj[:])
        names.append(col_name)
    return names

filepath = './UVA_' + str(sys.argv[1]) + '_vitals.mat'#'/Users/justinniestroy-admin/Desktop/NICU Vitals/'+ str(sys.argv[1])####UVA_' + str(sys.argv[1]) + '_vitals.mat'
#filepath = '/Users/justinniestroy-admin/Desktop/PreVent/UVA_2853_vitals.mat'
#print(filepath)
if len(sys.argv) > 2:
    print('diff')
    diff = True
else:
    diff = False
df,time = read_in_NICU_vitals(filepath)
time = time.to_numpy()
num_cols = df.shape[1]
time_series = {}
np.seterr(all='ignore')
for i in range(num_cols):
    if list(df.columns.values)[i] in ['HR']:#,'RESP','SPO2-%','SPO2-R']:
        time_series[list(df.columns.values)[i]] = df[list(df.columns.values)[i]].to_numpy()
        if diff:
            time_series[list(df.columns.values)[i]] = np.diff(time_series[list(df.columns.values)[i]])
    else:
        continue
if diff:
    time = time[1:]
interval_length = 60*10
step_size = 60*10
id = str(sys.argv[1])
if diff:
    id = id + '_diff'
print('ID to be saved to is: ' + str(id))
result = run.run_all(time_series,time,id)
