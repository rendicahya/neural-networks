import matplotlib.pyplot as plt
import numpy as np


def som(X_all, y_all, a, b, max_ep):
    cls, train_idx = np.unique(y_all, True)
    W = X_all[train_idx].astype(np.float64)
    X = np.array([x for i, x in enumerate(X_all) if i not in train_idx])
    y = np.array([y for i, y in enumerate(y_all) if i not in train_idx])
    hues = 'bo', 'go', 'ko', 'mo'
    dir = 'D:/LVQ'

    for i, c in enumerate(cls):
        x = X_all[y_all == c]
        plt.plot(x[:, 0], x[:, 1], hues[i])

    ep = 0

    plt.title('Data')
    plt.savefig('%s/0-0.png' % dir)

    plt.title('Bobot Awal')
    plt.plot(W[:, 0], W[:, 1], 'rX')
    plt.savefig('%s/0-1.png' % dir)
    plt.clf()

    while ep < max_ep:
        for i, x in enumerate(X):
            for i_, c in enumerate(cls):
                x_all = X_all[y_all == c]
                plt.plot(x_all[:, 0], x_all[:, 1], hues[i_])

            plt.title('Epoch: %d, data: %d' % (ep + 1, i + 1))
            plt.plot(W[:, 0], W[:, 1], 'rX')
            plt.plot(x[0], x[1], 'yD')
            plt.savefig('%s/%d-%da.png' % (dir, ep + 1, i + 1))
            plt.clf()

            d = [sum((w - x) ** 2) for w in W]
            min = np.argmin(d)
            s = 1 if y[i] == cls[min] else -1
            W[min] += s * a * (x - W[min])

            for i_, c in enumerate(cls):
                x_all = X_all[y_all == c]
                plt.plot(x_all[:, 0], x_all[:, 1], hues[i_])

            plt.title('Epoch: %d, data: %d' % (ep + 1, i + 1))
            plt.plot(W[:, 0], W[:, 1], 'rX')
            plt.savefig('%s/%d-%db.png' % (dir, ep + 1, i + 1))
            plt.clf()

        a *= b
        ep += 1

    return W


if __name__ == '__main__':
    x2 = np.array([[1.25, 1.25],
                   [1, 1],
                   [2, 1],
                   [1, 2],
                   [2, 2],
                   [3.25, 3.25],
                   [3, 3],
                   [4, 3],
                   [3, 4],
                   [4, 4]])

    x4 = np.array([[1.25, 3.25],
                   [1, 3],
                   [2, 3],
                   [1, 4],
                   [2, 4],
                   [3.25, 3.25],
                   [3, 3],
                   [4, 3],
                   [3, 4],
                   [4, 4],
                   [1.25, 1.25],
                   [1, 1],
                   [2, 1],
                   [1, 2],
                   [2, 2],
                   [3.25, 1.25],
                   [3, 1],
                   [4, 1],
                   [3, 2],
                   [4, 2]])

    y2 = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    y4 = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4])

    R = 0
    a = .6
    b = .5
    w = som(x4, y4, a, b, 3)

    print(w)
