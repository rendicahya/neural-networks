import numpy as np


def act(x):
    return [0 if i < 0 else i if 0 <= i <= 2 else 2 for i in x]


def run(x, r2, c1, c2, t_max):
    k = [c1] * (r2 * 2 + 1)
    k[0] = k[-1] = c2

    for t in range(t_max):
        x = act(np.convolve(x, k, 'same'))
        print(x)


if __name__ == '__main__':
    x = [-1, .5, .8, 1, .8, .5, -1]
    run(x, 2, .6, -.4, 5)
