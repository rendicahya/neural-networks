import numpy as np


def act(x):
    return 0 if x < 0 else x if 0 <= x <= 2 else 2


def iter(x, r2, c1, c2, t_max):
    k = [c1] * (r2 * 2 + 1)
    k[0] = k[-1] = c2

    for t in range(t_max):
        x_ = np.convolve(x, k, 'same')
        x = [act(i) for i in x_]
        print(x)


def main():
    x = [0, .5, .8, 1, .8, .5, 0]

    iter(x, 2, .6, -.4, 5)


if __name__ == '__main__':
    main()
