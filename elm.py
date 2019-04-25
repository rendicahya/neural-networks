import numpy as np


def elm_fit(X, t, h):
    W = np.array([[-.4, .2, .1],
                  [-.2, 0, .4],
                  [-.3, .3, -.1]])
    # W = np.random.uniform(-.5, .5, (h, len(X[0])))
    Hi = X @ W.T
    H = 1 / (1 + np.exp(-Hi))
    Ht = H.T
    Hp = np.linalg.inv(Ht @ H) @ Ht
    b = Hp @ t
    y = H @ b

    return W, b


def elm_predict():
    pass


def test():
    X = [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 0],
         [1, 1, 0],
         [0, 1, 0],
         [0, 0, 0],
         [0, 1, 0],
         [1, 1, 0],
         [0, 0, 0]]

    y = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    h = 3

    W, b = elm_fit(X, y, h)

    X = [[1, 0, 1],
         [1, 1, 0],
         [0, 1, 0]]


if __name__ == '__main__':
    test()
