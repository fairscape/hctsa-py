#© 2020 By The Rector And Visitors Of The University Of Virginia

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#import h5py
import numpy as np
#import multiprocessing as mp
import pandas as pd
#import make_operations
#operations = make_operations.make_operations()
#make_operations.make_otherfunctions()
from Operations import *
from Periphery import *
import new_run_algos as run
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
# def read_in_NICU_file(path):
#     arrays = {}
#     f = h5py.File(path,'r')
#     for k, v in f.items():
#         if k != 'vdata' and k != 'vt':
#             continue
#         arrays[k] = np.array(v)
#     df = pd.DataFrame(np.transpose(arrays['vdata']))
#     df = df.dropna(axis=1, how='all')
#     df.columns = get_column_names(f['vname'],f)
#     times = pd.Series(arrays['vt'][0], index=df.index)
#
#     return df,times

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

filepath = './NICU_' + str(sys.argv[1]) + '_vitals.mat'

if len(sys.argv) > 2:
    print('diff')
    diff = True
else:
    diff = False

df,time = read_NICU_vitals(filepath)
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
