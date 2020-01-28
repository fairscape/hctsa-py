import itertools
#import numba

#@numba.jit(nopython=True,parallel=True)
def EN_PermEn(y,m = 2,tau = 1):

    try:

        x = BF_embed(y,tau,m)

    except:

        return np.nan



    Nx = x.shape[0]

    permList = perms(m)
    numPerms = len(permList)


    countPerms = np.zeros(numPerms)

    for j in range(Nx):

        ix = np.argsort(x[j,:])

        for k in range(numPerms):


            if (permList[k,:] - ix == np.zeros(m)).all() :

                countPerms[k] = countPerms[k] + 1

                break

    p = countPerms / Nx

    p_0 = p[p > 0]

    permEn = -sum(np.multiply(p_0,np.log2(p_0)))

    mFact = math.factorial(m)

    normPermEn = permEn / np.log2(mFact)

    p_LE = np.maximum(np.repeat(1 / Nx,p.shape),p)

    permENLE = - np.sum(np.multiply(p_LE,np.log(p_LE))) / (m - 1)

    out = {'permEn':permEn,'normPermEn':normPermEn,'permEnLE':permENLE}

    return out

def perms(n):

    permut = itertools.permutations(np.arange(n))

    permut_array = np.empty((0,n))

    for p in permut:

        permut_array = np.append(permut_array,np.atleast_2d(p),axis=0)

    return(permut_array)
