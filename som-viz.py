import matplotlib.pyplot as plt
import numpy as np


def som(X, a, b, max_ep, W):
    W = np.random.uniform(1, 5, (W, len(X[0])))
    ep = 0
    dir = 'D:/SOM'

    plt.title('Epoch: 0')
    plt.plot(X[:, 0], X[:, 1], 'ko')
    plt.savefig('%s/0-0.png' % dir)
    plt.plot(W[:, 0], W[:, 1], 'bx')
    plt.savefig('%s/0-1.png' % dir)
    plt.clf()

    while ep < max_ep:
        for i, x in enumerate(X):
            plt.title('Epoch: %d, data: %d' % (ep + 1, i + 1))
            plt.plot(X[:, 0], X[:, 1], 'ko')
            plt.plot(W[:, 0], W[:, 1], 'bx')
            plt.plot(x[0], x[1], 'ro')
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
                  [4, 4],
                  [5, 4],
                  [4, 5],
                  [5, 5]])

    # w = np.array([[.2, .6, .5, .9],
    #               [.8, .4, .7, .3]])

    R = 0
    a = .6
    b = .5
    w = som(x, a, b, 5, 2)

    print(w)
