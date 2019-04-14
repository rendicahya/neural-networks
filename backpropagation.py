import numpy as np
from sklearn.datasets import load_iris


def sig(X):
    return [1 / (1 + np.exp(-x)) for x in X]


def sigd(X):
    for i, x in enumerate(X):
        s = sig([x])[0]

        yield s * (1 - s)


def bp_train(C, X, t, a, mep, mer):
    nin = [np.empty(c) for c in C]
    n = [np.empty(c + 1) if i < len(C) - 1 else np.empty(c) for i, c in enumerate(C)]
    w = [np.random.rand(C[i] + 1, C[i + 1]) for i in range(len(C) - 1)]
    dw = [np.empty((C[i] + 1, C[i + 1])) for i in range(len(C) - 1)]
    d = [np.empty(s) for s in C[1:]]
    din = [np.empty(s) for s in C[1:-1]]
    ep = 0
    er = []
    mse = 1

    w = np.array([[[.1, .2],
                   [.3, .4],
                   [.5, .6],
                   [.7, .8]],
                  [[.1, .4],
                   [.2, .5],
                   [.3, .6]]])

    for i in range(0, len(n) - 1):
        n[i][-1] = 1

    while ep < mep and mse > mer:
        ep += 1

        for r in range(len(X)):
            n[0][:-1] = X[r]

            for L in range(1, len(C)):
                nin[L] = np.dot(n[L - 1], w[L - 1])
                n[L][:len(nin[L])] = sig(nin[L])

            e = t[r] - n[-1]
            d[-1] = e * list(sigd(nin[-1]))
            dw[-1] = a * d[-1] * n[-2].reshape((-1, 1))

            for L in range(len(C) - 1, 1, -1):
                din[L - 2] = np.dot(d[L - 1], np.transpose(w[L - 1][:-1]))
                d[L - 2] = din[L - 2] * np.array(list(sigd(nin[L - 1])))
                dw[L - 2] = (a * d[L - 2]) * n[L - 2].reshape((-1, 1))

            w += dw

    return w


def bp_test():
    pass


r = 3, 2, 2
# iris = load_iris()
# X = iris.data
# y = iris.target
X = [[.8, .2, .1],
     [.1, .8, .9]]
y = [[0, 0],
     [0, 1]]
W = bp_train(r, X, y, .1, 1, .1)

print(W)
