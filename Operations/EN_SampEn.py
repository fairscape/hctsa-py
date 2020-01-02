import numpy as np
import warnings
#import numba

#@numba.jit(nopython=True,parallel=True)
def EN_SampEn(x,m=2,r=.2,scale=True):
    warnings.filterwarnings('ignore')
    if scale:
        r = np.std(x) * r

    templates = make_templates(x,m)
    #print(templates)
    A = 0
    B = 0
    for i in range(templates.shape[0]):
        template = templates[i,:]
        A = A + np.sum(np.amax(np.absolute(templates-template), axis=1) < r) -1
        B = B + np.sum(np.amax(np.absolute(templates[:,0:m]-template[0:m]),axis=1) < r) - 1
    if B == 0:
        return {'Sample Entropy':np.nan,"Quadratic Entropy":np.nan}
    
    return {'Sample Entropy':- np.log(A/B),"Quadratic Entropy": - np.log(A/B) + np.log(2*r)}
#@numba.jit(nopython=True,parallel=True)
def make_templates(x,m):
    N = int(len(x) - (m))
    templates = np.zeros((N,m+1))
    for i in range(N):
        templates[i,:] = x[i:i+m+1]
    return templates
# def EN_SampEn(y,M = 2,r = 0,pre = ''):
#     if r == 0:
#         r = .1*np.std(y)
#     else:
#         r = r*np.std(y)
#     M = M + 1
#     N = len(y)
#     print('hi')
#     lastrun = np.zeros(N)
#     run = np.zeros(N)
#     A = np.zeros(M)
#     B = np.zeros(M)
#     p = np.zeros(M)
#     e = np.zeros(M)
#
#     for i in range(1,N):
#         y1 = y[i-1]
#
#         for jj in range(1,N-i + 1):
#
#             j = i + jj - 1
#
#             if np.absolute(y[j] - y1) < r:
#
#                 run[jj] = lastrun[jj] + 1
#                 M1 = min(M,run[jj])
#                 for m in range(int(M1)):
#                     A[m] = A[m] + 1
#                     if j < N:
#                         B[m] = B[m] + 1
#             else:
#                 run[jj] = 0
#         for j in range(N-1):
#             lastrun[j] = run[j]
#
#     NN = N * (N - 1) / 2
#     p[0] = A[0] / NN
#     e[0] = - np.log(p[0])
#     for m in range(1,int(M)):
#         p[m] = A[m] / B[m-1]
#         e[m] = -np.log(p[m])
#     i = 0
#     # out = {'sampen':np.zeros(len(e)),'quadSampEn':np.zeros(len(e))}
#     # for ent in e:
#     #     quaden1 = ent + np.log(2*r)
#     #     out['sampen'][i] = ent
#     #     out['quadSampEn'][i] = quaden1
#     #     i = i + 1
#     out = {'Sample Entropy':e[1],'Quadratic Entropy':e[1] + np.log(2*r)}
#     return out
