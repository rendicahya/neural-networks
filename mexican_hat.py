import numpy as np


def act(x):
    return [0 if i < 0 else i if 0 <= i <= 2 else 2 for i in x]


def mexhat(x, r2, c1, c2, t_max):
    k = [c1] * (r2 * 2 + 1)
    k[0] = k[-1] = c2

    for t in range(t_max):
        print(x)
        x = act(np.convolve(x, k, 'same'))


if __name__ == '__main__':
    x = [0, .5, .8, 1, .8, .5, 0]
    mexhat(x, 2, .6, -.4, 3)
