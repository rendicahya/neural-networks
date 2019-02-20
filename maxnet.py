import numpy as np


def act(x):
    return x if x >= 0 else 0


def maxnet(a, e=None):
    if e is None:
        e = np.random.uniform(0, 1 / len(a))

    while np.count_nonzero(a) > 1:
        a_new = np.zeros(len(a))

        for i in range(len(a)):
            s = sum([a[j] for j in range(len(a)) if j != i])
            a_new[i] = act(a[i] - e * s)

        a = a_new

    return a


if __name__ == '__main__':
    a = [.2, .4, .6, .8]
    e = .2
    m = maxnet(a, e)

    print(m)
