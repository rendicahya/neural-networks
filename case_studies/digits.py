import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

from backpropagation import bp_fit


def bp():
    digits = datasets.load_digits()
    X = minmax_scale(digits.data)
    y = digits.target
    Y = np.zeros((len(X), 10), np.uint8)
    c = 64, 60, 10

    for i in range(len(y)):
        Y[i, y[i]] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3)
    w, ep, mse = bp_fit(c, X_train, y_train, .1, 1000, .1)


if __name__ == '__main__':
    bp()
