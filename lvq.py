import numpy as np


def lvq(X, y, a, b, max_ep):
    cls, train_idx = np.unique(y, True)
    W = X[train_idx].astype(np.float64)
    X = np.array([x for i, x in enumerate(X) if i not in train_idx])
    y = np.array([y for i, y in enumerate(y) if i not in train_idx])
    ep = 0

    while ep < max_ep:
        for i, x in enumerate(X):
            d = [sum((w - x) ** 2) for w in W]
            min = np.argmin(d)
            s = 1 if y[i] == cls[min] else -1
            W[min] += s * a * (x - W[min])

        a *= b
        ep += 1
        break

    return W


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

    # y = np.array([1, 1, 1, 1, 2, 2, 2, 2])

    a = .1
    b = .5
    w = lvq(X, y, a, b, 100)

    # print(w)
