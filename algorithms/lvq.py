import numpy as np


def lvq_fit(X, y, a, b, max_ep):
    c, train_idx = np.unique(y, True)
    W = X[train_idx].astype(np.float64)
    train = np.array([e for i, e in enumerate(zip(X, y)) if i not in train_idx])
    X = train[:, 0]
    y = train[:, 1]
    ep = 0

    while ep < max_ep:
        for i, x in enumerate(X):
            d = [sum((w - x) ** 2) for w in W]
            min = np.argmin(d)
            s = 1 if y[i] == c[min] else -1
            W[min] += s * a * (x - W[min])

        a *= b
        ep += 1

    return W, c


def lvq_predict(x, W):
    W, c = W
    d = [sum((w - x) ** 2) for w in W]

    return c[np.argmin(d)]


if __name__ == '__main__':
    X = np.array([[1, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 1],
                  [1, 0, 0, 0],
                  [0, 1, 1, 0]])

    y = np.array([1, 2, 2, 1, 2])

    # X = np.array([[1, 1],
    #               [2, 1],
    #               [1, 2],
    #               [2, 2],
    #               [3, 3],
    #               [4, 3],
    #               [3, 4],
    #               [4, 4]])
    #
    # y = np.array([1, 1, 1, 1, 2, 2, 2, 2])

    a = .1
    b = .5
    w = lvq_fit(X, y, a, b, 10)
    c = lvq_predict([1, 0, 1, 0], w)

    print(c)
