import sys

import matplotlib.pyplot as plt
import numpy as np


def line(w, th=0):
    w2 = w[2] + .001 if w[2] == 0 else w[2]

    return lambda x: (th - (w[1] * x) - w[0]) / w2


def plot(f, s, t):
    x = np.arange(-2, 3)
    col = 'ro', 'bo'

    for c, v in enumerate(np.unique(t)):
        p = s[np.where(t == v)]

        plt.plot(p[:, 1], p[:, 2], col[c])

    plt.axis([-2, 2, -2, 2])
    plt.plot(x, f(x))
    plt.show()


def ada_train(x, t, alpha=.1, max_err=.1, draw=False):
    w = np.random.uniform(0, 1, len(x[0]) + 1)
    b = np.ones((len(x), 1))
    x = np.hstack((b, x))
    stop = False
    ep = 0

    while not stop:
        ep += 1
        max_ch = -sys.maxsize

        print(f'\nEpoch #{ep}')

        for r, row in enumerate(x):
            y = np.dot(row, w)

            for i in range(len(row)):
                w_new = w[i] + alpha * (t[r] - y) * row[i]
                max_ch = max(abs(w[i] - w_new), max_ch)
                w[i] = w_new

            print(f'w: {w}')

            if draw:
                plot(line(w), x, t)

        stop = max_ch < max_err

    return w


def quiz():
    s = [[1, -1, -1],
         [-1, -1, -1],
         [-1, 1, 1]]
    t = [[-1], [1], [1]]
    ada_train(s, t, 1)


if __name__ == '__main__':
    quiz()
