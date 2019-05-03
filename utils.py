def bin_class(x):
    length = len(bin(max(x))[2:])

    for i in x:
        b = bin(i)[2:].zfill(length)

        yield [int(n) for n in b]


def test():
    x1 = bin_class([1, 2, 3])
    x2 = bin_class([1, 2, 3, 4])

    print(list(x1))
    print(list(x2))


if __name__ == '__main__':
    test()
