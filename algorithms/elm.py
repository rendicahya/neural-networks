import numpy as np


def elm_fit(X, t, h):
    W = np.random.uniform(-.5, .5, (h, len(X[0])))
    Hi = X @ W.T
    H = 1 / (1 + np.exp(-Hi))
    Ht = H.T
    Hp = np.linalg.inv(Ht @ H) @ Ht
    b = Hp @ t
    y = H @ b
    mape = sum(abs(y - t) / t * 100) / len(t)

    return W, b, mape


def elm_predict(X, W, b):
    Hi = X @ W.T
    H = 1 / (1 + np.exp(-Hi))
    y = H @ b

    return y


def test():
    X_train = [[1, 1, 1],
               [1, 0, 1],
               [1, 1, 0],
               [1, 1, 0],
               [0, 1, 0],
               [0, 0, 0],
               [0, 1, 0],
               [1, 1, 0],
               [0, 0, 0]]

    y_train = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    h = 3

    W, b, mape = elm_fit(X_train, y_train, h)

    X_test = [[1, 0, 1],
              [1, 1, 0],
              [0, 1, 0]]

    y_test = [1, 1, 3]
    y_predict = elm_predict(X_test, W, b)
    mape = sum(abs(y_predict - y_test) / y_test * 100) / len(y_test)

    print(y_test)
    print(y_predict)
    print(mape)


if __name__ == '__main__':
    test()
