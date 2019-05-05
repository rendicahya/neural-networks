import numpy as np

from algorithms.maxnet import maxnet


def hamming(ex, x):
    w = np.array(ex) / 2
    b = len(w[0]) / 2
    y = [b + sum(x * w[i]) for i in range(len(w))]
    m = maxnet(y)

    return ex[m]


if __name__ == '__main__':
    ex = [[1, -1, -1, -1],
          [-1, -1, -1, 1]]
    x = [1, 1, -1, -1]
    h = hamming(ex, x)

    print(h)
