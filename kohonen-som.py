import numpy as np


def som(X, W, max_ep=100):
    ep = 0

    while ep < max_ep:
        for x in X:
            for c, w in enumerate(W):
                print(sum((w - x) ** 2))

            print()

        break


if __name__ == '__main__':
    x = np.array([[1, 1, 0, 0],
                  [0, 0, 0, 1],
                  [1, 0, 0, 0],
                  [0, 0, 1, 1]])

    w = np.array([[.2, .6, .5, .9],
                  [.8, .4, .7, .3]])

    R = 0
    a = .6
    som(x, w, 1)
