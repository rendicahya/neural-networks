import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

from draw_utils import line, plot


def step(y, th=0.):
    return 1 if y > th else 0 if -th <= y <= th else -1


def percep_train(s: list, t: list, th: float = 0., a: float = 1, draw: bool = False):
    w = np.zeros(len(s[0]) + 1)
    b = np.ones((len(s), 1))
    s = np.hstack((b, s))
    stop = False
    epoch = 0

    while not stop:
        stop = True
        epoch += 1

        for r, row in enumerate(s):
            y_in = np.dot(row, w)
            y = step(y_in, th)

            if y != t[r]:
                stop = False
                w = [w[i] + a * t[r] * row[i] for i in range(len(row))]

            if draw:
                plot([line(w, th), line(w, -th)], s, t)

    return w, epoch


def percep_test(X, w, th=0):
    for x in X:
        y_in = w[0] + np.dot(x, w[1:])

        yield step(y_in, th)


def test():
    # s = [[1, -1, 1],
    #      [1, -1, 1],
    #      [-1, 1, -1],
    #      [-1, -1, 1]]
    # t = [1, 1, -1, -1]

    s = [[1, -1, -1],
         [-1, -1, -1],
         [-1, 1, -1]]
    t = [-1, 1, 1]

    w = percep_train(s, t, .5)
    y = percep_test([-1, 1, -1], w)

    print(y)


def test_iris():
    iris = sns.load_dataset('iris')
    iris = iris.loc[iris['species'] != 'virginica']
    iris = iris.drop(['sepal_width', 'petal_width'], axis=1)

    iris.loc[iris['species'] == 'setosa', 'species'] = -1
    iris.loc[iris['species'] == 'versicolor', 'species'] = 1

    X = iris[['sepal_length', 'petal_length']].to_numpy()
    X = minmax_scale(X)
    y = iris['species'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

    w, epoch = percep_train(X_train, y_train)
    res = list(percep_test(X_test, w))
    acc = accuracy_score(res, y_test)

    print(f'Epoch: {epoch}')
    print(f'Accuracy: {acc}')

    # sns.pairplot(iris, hue='species')
    # plt.show()


if __name__ == '__main__':
    test_iris()
