import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

from draw_utils import line, plot


def step(y, th=0.):
    return 1 if y > th else 0 if -th <= y <= th else -1


def percep_train(s, t, th=0., a=1, draw=None):
    nx = len(s[0])
    ny = len(t[0])
    w = np.zeros((nx + 1, ny))
    b = np.ones((len(s), 1))
    s = np.hstack((s, b))
    stop = False
    epoch = 0
    # w = np.arange(1, 9).reshape((4, 2))

    while not stop:
        stop = True
        epoch += 1
        print(f'epoch: {epoch}')

        for i, s_ in enumerate(s):
            for j, t_ in enumerate(t[i]):
                yin = np.dot(s_, w[:, j:j + 1])[0]
                y = step(yin, th)

                if y != t_:
                    stop = False
                    dw = a * t_ * s_
                    w[:, j:j + 1] += dw.reshape(nx + 1, -1)

                # if draw == 'loop':
                #     plot([line(w, th), line(w, -th)], s, t)

        # print(w)
        # if draw == 'result':
        #     plot([line(w, th), line(w, -th)], s, t)

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

    # s = [[1, -1, -1],
    #      [-1, -1, -1],
    #      [-1, 1, -1]]
    # t = [-1, 1, 1]

    s = [[.8, .8, .8],
         [.2, .2, .2]]
    t = [[1, 0],
         [0, 1]]

    w = percep_train(s, t, .5)
    # y = percep_test([-1, 1, -1], w)

    # print(y)


def test_iris():
    iris = sns.load_dataset('iris')
    iris = iris.loc[iris['species'] != 'virginica']
    iris = iris.drop(['sepal_width', 'petal_width'], axis=1)

    # iris.loc[iris['species'] == 'virginica', 'species'] = 0
    # iris.loc[iris['species'] == 'setosa', 'species'] = 1
    # iris.loc[iris['species'] == 'versicolor', 'species'] = 2

    X = iris[['sepal_length', 'petal_length']].to_numpy()
    X = minmax_scale(X)
    c = {'virginica': [1, 0, 0], 'setosa': [0, 1, 0], 'versicolor': [0, 0, 1]}
    y = [c[i] for i in iris['species'].to_numpy()]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

    print(y)

    w, epoch = percep_train(X_train, y_train)
    # res = list(percep_test(X_test, w))
    # acc = accuracy_score(res, y_test)

    # print(f'Epoch: {epoch}')
    # print(f'Accuracy: {acc}')


if __name__ == '__main__':
    test()
