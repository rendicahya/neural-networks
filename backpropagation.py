import numpy as np


def sig(X):
    return [1 / (1 + np.exp(-x)) for x in X]


def sigd(X):
    for i, x in enumerate(X):
        s = sig([x])[0]

        yield s * (1 - s)


def bp_fit(C, X, t, a, mep, mer):
    nin = [np.empty(i) for i in C]
    n = [np.empty(j + 1) if i < len(C) - 1 else np.empty(j) for i, j in enumerate(C)]
    w = np.array([np.random.rand(C[i] + 1, C[i + 1]) for i in range(len(C) - 1)])
    dw = [np.empty((C[i] + 1, C[i + 1])) for i in range(len(C) - 1)]
    d = [np.empty(s) for s in C[1:]]
    din = [np.empty(s) for s in C[1:-1]]
    ep = 0
    mse = 1

    for i in range(0, len(n) - 1):
        n[i][-1] = 1

    while ep < mep and mse > mer:
        ep += 1
        mse = 0

        for r in range(len(X)):
            n[0][:-1] = X[r]

            for L in range(1, len(C)):
                nin[L] = np.dot(n[L - 1], w[L - 1])

                n[L][:len(nin[L])] = sig(nin[L])

            e = t[r] - n[-1]
            mse += sum(e ** 2)
            d[-1] = e * list(sigd(nin[-1]))
            dw[-1] = a * d[-1] * n[-2].reshape((-1, 1))

            for L in range(len(C) - 1, 1, -1):
                din[L - 2] = np.dot(d[L - 1], np.transpose(w[L - 1][:-1]))
                d[L - 2] = din[L - 2] * np.array(list(sigd(nin[L - 1])))
                dw[L - 2] = (a * d[L - 2]) * n[L - 2].reshape((-1, 1))

            w += dw

        mse /= len(X)

    return w, ep, mse


def bp_predict(X, w):
    n = [np.empty(len(i)) for i in w]
    nin = [np.empty(len(i[0])) for i in w]

    n.append(np.empty(len(w[-1][0])))

    for x in X:
        n[0][:-1] = x

        for L in range(0, len(w)):
            nin[L] = np.dot(n[L], w[L])
            n[L + 1][:len(nin[L])] = sig(nin[L])

        yield n[-1].copy()


def test3layer():
    c = 3, 2, 2
    X = [[.8, .2, .1],
         [.1, .8, .9]]
    y = [[0, 0],
         [0, 1]]
    w = np.array([[[.1, .2],
                   [.3, .4],
                   [.5, .6],
                   [.7, .8]],
                  [[.1, .4],
                   [.2, .5],
                   [.3, .6]]])

    w, ep, mse = bp_fit(c, X, y, .1, 100, .1, w)
    y = bp_predict([.8, .2, .1], w)

    print(y)


def test5layer():
    pass
    # w = np.array([[[.1, .2],
    #                [.3, .4],
    #                [.5, .6],
    #                [.7, .8]],
    #               [[.1, .4],
    #                [.2, .5],
    #                [.3, .6]],
    #               [[.3, .4, .5],
    #                [.6, .5, .4],
    #                [.8, .7, .6]]])


if __name__ == '__main__':
    test3layer()
