from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

from algorithms.backpropagation import bp_fit
from algorithms.elm import elm_fit, elm_predict
from utils.label_encoding import onehot_enc, onehot_dec


def bp():
    digits = datasets.load_digits()
    X = minmax_scale(digits.data)
    Y = onehot_enc(digits.target)
    c = 64, 50, 10

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3)
    w, ep, mse = bp_fit(c, X_train, y_train, .1, -1, .1)


def elm():
    digits = datasets.load_digits()
    X = minmax_scale(digits.data)
    Y = onehot_enc(digits.target)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3)
    W, b, mape = elm_fit(X_train, y_train, 70)

    print(f'MAPE: {mape}')

    predict = elm_predict(X_test, W, b)
    predict = onehot_dec(predict)
    y_test = onehot_dec(y_test)
    acc = accuracy_score(predict, y_test)

    print(f'Output: {predict}')
    print(f'True  : {y_test}')
    print(f'Accuracy: {acc}')


if __name__ == '__main__':
    elm()
