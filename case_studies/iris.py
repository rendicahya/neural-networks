import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

from backpropagation import bp_predict, bp_fit
from elm import elm_fit, elm_predict
from utils.label_pattern import bin_enc, bin_dec


def plot():
    iris = sns.load_dataset('iris')
    # iris = iris.loc[iris['species'] != 'virginica']
    iris = iris.drop(['sepal_width', 'petal_width'], axis=1)
    iris.loc[iris['species'] == 'virginica', 'petal_length'] += 2

    # X = iris[['sepal_length', 'petal_length']].to_numpy()
    # X = minmax_scale(X)

    # y = iris['species'].to_numpy()
    # c = {'setosa': 0, 'versicolor': 1}
    # y = [c[i] for i in y]
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

    sns.pairplot(iris, hue='species')
    plt.show()


def bp():
    iris = load_iris()
    X = minmax_scale(iris.data)
    Y = bin_enc(iris.target)

    c = 4, 3, 2
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3)
    w, ep, mse = bp_fit(c, X_train, y_train, .1, 1000, .1)

    print(f'Epoch: {ep}')
    print(f'MSE: {mse}')

    predict = bp_predict(X_test, w)
    predict = bin_dec(predict)
    y_test = bin_dec(y_test)
    acc = accuracy_score(predict, y_test)

    print(f'Output: {predict}')
    print(f'True  : {y_test}')
    print(f'Accuracy: {acc}')


def elm():
    iris = load_iris()
    X = minmax_scale(iris.data)
    Y = bin_enc(iris.target)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3)
    W, b, mape = elm_fit(X_train, y_train, 5)

    print(f'MAPE: {mape}')

    predict = elm_predict(X_test, W, b)
    predict = bin_dec(predict)
    y_test = bin_dec(y_test)
    acc = accuracy_score(predict, y_test)

    print(f'Output: {predict}')
    print(f'True  : {y_test}')
    print(f'Accuracy: {acc}')


if __name__ == '__main__':
    plot()
