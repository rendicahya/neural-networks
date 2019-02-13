def binstep(x, th=0):
    return 1 if x >= th else 0


def bipstep(y, th=0):
    return 1 if y >= th else -1


def percep_step(y, th=0):
    return 1 if y > th else 0 if -th <= y <= th else -1
