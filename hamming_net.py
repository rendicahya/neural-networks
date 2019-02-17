import numpy as np


def run(ex, x):
    w = np.array(ex) / 2
    b = len(w[0]) / 2

    return [b + sum(x * w[i]) for i in range(len(w))]


def main():
    ex = [[1, -1, -1, -1],
          [-1, -1, -1, 1]]
    x = [1, 1, -1, -1]
    h = run(ex, x)

    print(h)


if __name__ == '__main__':
    main()
