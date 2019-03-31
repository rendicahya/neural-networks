import matplotlib.pyplot as plt
import numpy as np


def som(X, y, a, b, max_ep, W):
    hues = 'bo', 'go', 'ko'
    cls = np.unique(y)
    dir = 'D:/LVQ'

    for i, c in enumerate(cls):
        x = X[y == c]
        plt.plot(x[:, 0], x[:, 1], hues[i])

    W = np.array([X[y == c][0] for c in cls])
    ep = 0

    plt.title('Data')
    plt.savefig('%s/0-0.png' % dir)

    plt.title('Bobot Awal')
    plt.plot(W[:, 0], W[:, 1], 'rX')
    plt.savefig('%s/0-1.png' % dir)
    plt.clf()

    while ep < max_ep:
        for i, x in enumerate(X):
            for i, c in enumerate(cls):
                x = X[y == c]
                plt.plot(x[:, 0], x[:, 1], hues[i])

            plt.title('Epoch: %d, data: %d' % (ep + 1, i + 1))
            plt.plot(W[:, 0], W[:, 1], 'rX')
            plt.savefig('%s/%d-%da.png' % (dir, ep + 1, i + 1))
            plt.clf()

            d = [sum((w - x) ** 2) for w in W]
            min = np.argmin(d)
            W[min] += a * (x - W[min])

            plt.title('Epoch: %d, data: %d' % (ep + 1, i + 1))
            plt.plot(X[:, 0], X[:, 1], 'ko')
            plt.plot(W[:, 0], W[:, 1], 'bx')
            plt.plot(x[0], x[1], 'ro')
            plt.savefig('%s/%d-%db.png' % (dir, ep + 1, i + 1))
            plt.clf()

        a *= b
        ep += 1

    return W


if __name__ == '__main__':
    # x = np.array([[1, 1, 0, 0],
    #               [0, 0, 0, 1],
    #               [1, 0, 0, 0],
    #               [0, 0, 1, 1]])

    x = np.array([[1, 1],
                  [2, 1],
                  [1, 2],
                  [2, 2],
                  [3, 3],
                  [4, 3],
                  [3, 4],
                  [4, 4]])

    y = np.array([1, 1, 1, 1, 2, 2, 2, 2])

    # w = np.array([[.2, .6, .5, .9],
    #               [.8, .4, .7, .3]])

    R = 0
    a = .6
    b = .5
    w = som(x, y, a, b, 5, 2)

    print(w)
