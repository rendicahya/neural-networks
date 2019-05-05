import numpy as np


def bin_enc(lbl):
    mi = min(lbl)
    length = len(bin(max(lbl) - mi + 1)[2:])
    enc = []

    for i in lbl:
        b = bin(i - mi)[2:].zfill(length)

        enc.append([int(n) for n in b])

    return enc


def bin_dec(enc, mi=0):
    lbl = []

    for e in enc:
        rounded = [int(round(x)) for x in e]
        string = ''.join(str(x) for x in rounded)
        num = int(string, 2) + mi

        lbl.append(num)

    return lbl


def onehot_enc(lbl, min_val=0):
    mi = min(lbl)
    enc = np.full((len(lbl), max(lbl) - mi + 1), min_val, np.int8)

    for i, x in enumerate(lbl):
        enc[i, x - mi] = 1

    return enc


def onehot_dec(enc, mi=0):
    return [np.argmax(e) + mi for e in enc]


def test():
    # print(bin_enc([1, 2, 3]))
    print(bin_enc([1, 2, 3, 4]))

    # print(bin_dec([[0, 0], [0, 1], [1, 0]], 1))

    # print(to_class([[.1, .9, .1]]))
    # print(to_class([[0.20232853, 0.67238648]]))
    #
    # print(onehot_enc([1, 2, 3]))
    # print(onehot_enc([3, 4, 5, 6]))
    # print(onehot_enc([2, 3]))

    # print(onehot_dec([[1, 0, 0, 0],
    #                   [0, 1, 0, 0],
    #                   [0, 0, 1, 0],
    #                   [0, 0, 0, 1]], 3))


if __name__ == '__main__':
    test()
