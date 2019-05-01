import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

from backpropagation import bp_predict, bp_fit


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


def classify():
    c = 4, 3, 2

    iris = load_iris()
    X = minmax_scale(iris.data)

    p = np.array([[0, 0], [0, 1], [1, 0]])

    X_train, X_test, y_train, y_test = train_test_split(X, iris.target, test_size=.3)
    w, ep, mse = bp_fit(c, X_train, [p[i] for i in y_train], .1, 1000, .1)

    print(f'Epoch: {ep}')
    print(f'MSE: {mse}')

    out = list(bp_predict(X_test, w))
    out = [np.argmin(np.sum(abs(i - p), axis=1)) for i in out]
    acc = accuracy_score(out, y_test)

    print(f'Output: {out}')
    print(f'True  : {y_test}')
    print(f'Accuracy: {acc}')


if __name__ == '__main__':
    classify()
