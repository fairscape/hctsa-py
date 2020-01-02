import numpy as np
import math


def rm_information(*args):
    """
    rm_information estimates the mutual information of the two stationary signals with
    independent pairs of samples using various approaches:

    takes in between 2 and 5 parameters:
        rm_information(x, y)
        rm_information(x, y, descriptor)
        rm_information(x, y, descriptor, approach)
        rm_information(x, y, descriptor, approach, base)

    :returns estimate, nbias, sigma, descriptor

        estimate : the mututal information estimate
        nbias : n-bias of the estimate
        sigma : the standard error of the estimate
        descriptor : the descriptor of the histogram, see also rm_histogram2

            lowerbound? : lowerbound of the histogram in the ? direction
            upperbound? : upperbound of the histogram in the ? direction
            ncell? : number of cells in the histogram in ? direction

        approach : method used, choose from the following:

            'unbiased'  : the unbiased estimate (default)
            'mmse'      : minimum mean square estimate
            'biased'    : the biased estimate

        base : the base of the logarithm, default e

    MATLAB function and logic by Rudy Moddemeijer
    Translated to python by Tucker Cullen
    """

    nargin = len(args)

    if nargin < 1:
        print("Takes in 2-5 parameters: ")
        print("rm_information(x, y)")
        print("rm_information(x, y, descriptor)")
        print("rm_information(x, y, descriptor, approach)")
        print("rm_information(x, y, descriptor, approach, base)")
        print()

        print("Returns a tuple containing: ")
        print("estimate, nbias, sigma, descriptor")
        return

    # some initial tests on the input arguments

    x = np.array(args[0])  # make sure the imputs are in numpy array form
    y = np.array(args[1])

    xshape = x.shape
    yshape = y.shape

    lenx = xshape[0]  # how many elements are in the row vector
    leny = yshape[0]

    if len(xshape) != 1:  # makes sure x is a row vector
        print("Error: invalid dimension of x")
        return

    if len(yshape) != 1:
        print("Error: invalid dimension of y")
        return

    if lenx != leny:  # makes sure x and y have the same amount of elements
        print("Error: unequal length of x and y")
        return

    if nargin > 5:
        print("Error: too many arguments")
        return

    if nargin < 2:
        print("Error: not enough arguments")
        return

    # setting up variables depending on amount of inputs

    if nargin == 2:
        hist = rm_histogram2(x, y)  # call outside function from rm_histogram2.py
        h = hist[0]
        descriptor = hist[1]

    if nargin >= 3:
        hist = rm_histogram2(x, y, args[2])  # call outside function from rm_histogram2.py, args[2] represents the given descriptor
        h = hist[0]
        descriptor = hist[1]

    if nargin < 4:
        approach = 'unbiased'
    else:
        approach = args[3]

    if nargin < 5:
        base = math.e  # as in e = 2.71828
    else:
        base = args[4]

    lowerboundx = descriptor[0, 0]  #not sure why most of these were included in the matlab script, most of them go unused
    upperboundx = descriptor[0, 1]
    ncellx = descriptor[0, 2]
    lowerboundy = descriptor[1, 0]
    upperboundy = descriptor[1, 1]
    ncelly = descriptor[1, 2]

    estimate = 0
    sigma = 0
    count = 0

    # determine row and column sums

    hy = np.sum(h, 0)
    hx = np.sum(h, 1)

    ncellx = ncellx.astype(int)
    ncelly = ncelly.astype(int)

    for nx in range(0, ncellx):
        for ny in range(0, ncelly):
            if h[nx, ny] != 0:
                logf = math.log(h[nx, ny] / hx[nx] / hy[ny])
            else:
                logf = 0

            count = count + h[nx, ny]
            estimate = estimate + h[nx, ny] * logf
            sigma = sigma + h[nx, ny] * (logf ** 2)

    # biased estimate

    estimate = estimate / count
    sigma = math.sqrt((sigma / count - estimate ** 2) / (count - 1))
    estimate = estimate + math.log(count)
    nbias = (ncellx - 1) * (ncelly - 1) / (2 * count)

    # conversion to unbiased estimate

    if approach[0] == 'u':
        estimate = estimate - nbias
        nbias = 0

        # conversion to minimum mse estimate

    if approach[0] == 'm':
        estimate = estimate - nbias
        nbias = 0
        lamda = (estimate ** 2) / ((estimate ** 2) + (sigma ** 2))
        nbias = (1 - lamda) * estimate
        estimate = lamda * estimate
        sigma = lamda * sigma

        # base transformations

    estimate = estimate / math.log(base)
    nbias = nbias / math.log(base)
    sigma = sigma / math.log(base)

    return estimate, nbias, sigma, descriptor
