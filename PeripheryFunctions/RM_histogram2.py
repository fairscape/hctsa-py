
import numpy as np
import math


def rm_histogram2( *args ):
    """
    rm_histogram2() computes the two dimensional frequency histogram of two row vectors x and y

    Takes in either two or three parameters:
        rm_histogram(x, y)
        rm_histogram(x, y, descriptor)

    x, y : the row vectors to be analyzed
    descriptor : the descriptor of the histogram where:

        descriptor = [lowerx, upperx, ncellx, lowery, uppery, ncelly]
            lower? : the lowerbound of the ? dimension of the histogram
            upper? : the upperbound of the dimension of the histogram
            ncell? : the number of cells of the ? dimension of the histogram

    :return: a tuple countaining a) the result (the 2d frequency histogram), b) descriptor (the descriptor used)

    MATLAB function and logic by Rudy Moddemeijer
    Translated to python by Tucker Cullen

    """

    nargin = len(args)

    if nargin < 1:
        print("Usage: result = rm_histogram2(X, Y)")
        print("       result = rm_histogram2(X,Y)")
        print("Where: descriptor = [lowerX, upperX, ncellX; lowerY, upperY, ncellY")

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

    if nargin > 3:
        print("Error: too many arguments")
        return

    if nargin == 2:
        minx = np.amin(x)
        maxx = np.amax(x)
        deltax = (maxx - minx) / (lenx - 1)
        ncellx = math.ceil(lenx ** (1 / 3))

        miny = np.amin(y)
        maxy = np.amax(y)
        deltay = (maxy - miny) / (leny - 1)
        ncelly = ncellx
        descriptor = np.array(
            [[minx - deltax / 2, maxx + deltax / 2, ncellx], [miny - deltay / 2, maxy + deltay / 2, ncelly]])
    else:
        descriptor = args[2]

    lowerx = descriptor[0, 0]  # python indexes one less then matlab indexes, since starts at zero
    upperx = descriptor[0, 1]
    ncellx = descriptor[0, 2]
    lowery = descriptor[1, 0]
    uppery = descriptor[1, 1]
    ncelly = descriptor[1, 2]

    # checking descriptor to make sure it is valid, otherwise print an error

    if ncellx < 1:
        print("Error: invalid number of cells in X dimension")

    if ncelly < 1:
        print("Error: invalid number of cells in Y dimension")

    if upperx <= lowerx:
        print("Error: invalid bounds in X dimension")

    if uppery <= lowery:
        print("Error: invalid bounds in Y dimension")

    result = np.zeros([int(ncellx), int(ncelly)],
                      dtype=int)  # should do the same thing as matlab: result(1:ncellx,1:ncelly) = 0;

    xx = np.around((x - lowerx) / (upperx - lowerx) * ncellx + 1 / 2)
    yy = np.around((y - lowery) / (uppery - lowery) * ncelly + 1 / 2)

    xx = xx.astype(int)  # cast all the values in xx and yy to ints for use in indexing, already rounded in previous step
    yy = yy.astype(int)

    for n in range(0, lenx):
        indexx = xx[n]
        indexy = yy[n]

        indexx -= 1  # adjust indices to start at zero, not one like in MATLAB
        indexy -= 1

        if indexx >= 0 and indexx <= ncellx - 1 and indexy >= 0 and indexy <= ncelly - 1:
            result[indexx, indexy] = result[indexx, indexy] + 1

    return result, descriptor
