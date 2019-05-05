import numpy as np


def binstep(y, th=0):
    return [1 if i > th else 0 for i in y]


def hebb_assoc_train(s, t):
    w = np.zeros((len(s[0]), len(t[0])))

    for r, row in enumerate(s):
        for c, col in enumerate(row):
            w[c] = [w[c, i] + col * t[r, i] for i in range(len(t[r]))]

    return w


def hebb_assoc_train_mat(s, t):
    p = [np.outer(np.reshape(s[i], (-1, 1)), t[i]) for i in range(len(s))]

    return np.sum(p, 0)


def hebb_assoc_test(x, w):
    y = [np.dot(x, w[:, i]) for i in range(len(w[0]))]

    return binstep(y)


def hebb_assoc_test_mat(x, w):
    return binstep(np.matmul(x, w))


if __name__ == '__main__':
    s = [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 1]]

    t = np.array([[1, 0],
                  [1, 0],
                  [0, 1],
                  [0, 1]])

    w = hebb_assoc_train_mat(s, t)
    y = hebb_assoc_test([1, 1, 0, 0], w)

    print(w)
    print(y)
