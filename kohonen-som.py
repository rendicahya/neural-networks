import numpy as np


def som(X, a, b, max_ep, W):
    W = np.random.uniform(size=(W, len(X[0])))
    ep = 0

    while ep < max_ep:
        for x in X:
            d = [sum((w - x) ** 2) for w in W]
            min = np.argmin(d)
            W[min] += a * (x - W[min])

        a *= b
        ep += 1

    return W


if __name__ == '__main__':
    x = np.array([[1, 1, 0, 0],
                  [0, 0, 0, 1],
                  [1, 0, 0, 0],
                  [0, 0, 1, 1]])

    # w = np.array([[.2, .6, .5, .9],
    #               [.8, .4, .7, .3]])

    R = 0
    a = .6
    b = .5
    w = som(x, a, b, 100, 2)

    print(w)
