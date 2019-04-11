import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def bp_train(arch):
    w1 = np.random.rand(arch[0], arch[1])
    dw1 = np.random.rand(arch[0], arch[1])

    w2 = np.random.rand(arch[1], arch[2])
    dw2 = np.random.rand(arch[1], arch[2])

    in1 = np.zeros(arch[0], np.float64)
    in2 = np.zeros(arch[1], np.float64)
    in3 = np.zeros(arch[2], np.float64)


def bp_test():
    pass


bp_train((2, 3, 2))
