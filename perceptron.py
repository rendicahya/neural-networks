import numpy as np

from draw_utils import line, plot


def step(y, th=0):
    return 1 if y > th else 0 if -th <= y <= th else -1


def percep_train(s, t, th=0, a=1, draw=False):
    w = np.zeros(len(s[0]) + 1)
    b = np.ones((len(s), 1))
    s = np.hstack((b, s))
    stop = False
    epoch = 0

    while not stop:
        stop = True
        epoch += 1

        print('\nEpoch #%d' % epoch)

        for r, row in enumerate(s):
            y_in = np.dot(row, w)
            y_out = step(y_in, th)

            if y_out != t[r]:
                stop = False
                w = [w[i] + a * t[r] * row[i] for i in range(len(row))]

            print('Bobot: {}'.format(w))

            if draw:
                plot(line(w, th), line(w, -th), s, t)

    return w
