
import random

def BF_ResetSeed(resetHow= 'default'):
    '''

    Allows functions using random numbers to produce repeatable results with a consistent syntax

    :param resetHow = method for resetting the seed
    :return: void
    '''


    if resetHow == 'default':

        random.seed(0) # resets to using Marsenne Twister method with seed of 0

    elif resetHow == 'none': # don't change the seed
        return

    else:
        print("Error: Not sure how to reset with resetHow = " + resetHow)
        return
