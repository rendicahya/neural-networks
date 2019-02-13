import numpy as np

from draw_utils import line, plot
from functions import percep_step


def percep_train(s, t, th=0, alpha=1, draw=False):
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
            y_out = percep_step(y_in, th)

            if y_out != t[r]:
                stop = False
                w = [w[i] + alpha * t[r] * row[i] for i in range(len(row))]

            print('Bobot: {}'.format(w))

            if draw:
                plot(line(w, th), line(w, -th), s, t)

    return w
