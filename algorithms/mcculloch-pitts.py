import numpy as np

from utils.functions import binstep


def AND(x):
    y_in = np.dot(x, [1, 1])

    return binstep(y_in, 2)


def OR(x):
    y_in = np.dot(x, [2, 2])

    return binstep(y_in, 2)


def ANDNOT(x):
    y_in = np.dot(x, [2, -1])

    return binstep(y_in, 2)


def XOR(x):
    z = [ANDNOT(x), ANDNOT(x[::-1])]

    return OR(z)
