def to_pattern(X):
    bin_length = len(bin(max(X))[2:])
    result = []

    for i in X:
        b = bin(i)[2:].zfill(bin_length)

        result.append([int(n) for n in b])

    return result


def to_class(pattern):
    classes = []

    for p in pattern:
        rounded = [int(round(x)) for x in p]
        string = ''.join(str(x) for x in rounded)
        integer = int(string, 2)

        classes.append(integer)

    return classes


def test():
    print(to_pattern([1, 2, 3]))
    print(to_pattern([1, 2, 3, 4]))
    print(to_class([.1, .9, .1]))
    print(to_class([0.20232853, 0.67238648]))


if __name__ == '__main__':
    test()
