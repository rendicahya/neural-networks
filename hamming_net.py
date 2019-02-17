import numpy as np
import maxnet


def run(ex, x):
    w = np.array(ex) / 2
    b = len(w[0]) / 2
    y = [b + sum(x * w[i]) for i in range(len(w))]
    m = maxnet.run(y)

    return m


def main():
    ex = [[1, -1, -1, -1],
          [-1, -1, -1, 1]]
    x = [1, 1, -1, -1]
    h = run(ex, x)

    print(h)


if __name__ == '__main__':
    main()
