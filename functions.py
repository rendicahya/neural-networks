def binstep(x, th=0) -> int:
    """
    Thresholding a value either to 1 or 0 based on the threshold value.

    :param x: The input value
    :param th: The threshold value
    :return: The thresholding result (0 or 1)
    """
    return 1 if x >= th else 0


def bipstep(x, th=0) -> int:
    """
    Thresholding a value either to 1 or -1 based on the threshold value.

    :param x: The input value
    :param th: The threshold value
    :return: The thresholding result (-1 or 1)
    """
    return 1 if x >= th else -1
