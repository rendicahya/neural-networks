import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale


def step(x, th):
    return 1 if x > th else 0 if -th <= x <= th else -1


def percep_fit(s, t, th=0., a=1, draw=False):
    nx = len(s[0])
    ny = len(t[0])
    w = np.zeros((nx + 1, ny))
    b = np.ones((len(s), 1))
    s = np.hstack((s, b))
    stop = False
    epoch = 0

    while not stop:
        stop = True
        epoch += 1

        print(f'Epoch: {epoch}')

        for i, s_ in enumerate(s):
            for j, t_ in enumerate(t[i]):
                yin = np.dot(s_, w[:, j:j + 1])[0]
                y = step(yin, th)

                print(f'yin: {yin}')
                print(f'y: {y}')
                print(f'w: {w}')

                if y != t_:
                    stop = False
                    dw = a * t_ * s_
                    w[:, j:j + 1] += dw.reshape(nx + 1, -1)

                print(f'w\': {w}')

                # if draw == 'loop' and nx == 2:
                #     plot([line(w, th), line(w, -th)], s, t)

        stop = True

        # if draw == 'result' and nx == 2:
        #     plot([line(w, th), line(w, -th)], s, t)

    return w, epoch


def percep_predict(X, w, th=0):
    for x in X:
        y_in = np.dot([*x, 1], w)
        y = [step(i, th) for i in y_in]

        yield y


def quiz():
    s = [[1, -1, 1],
         [-1, -1, -1],
         [-1, 1, -1]]
    t = [[-1], [1], [1]]
    percep_fit(s, t, .5, 1)


def test():
    s = [[1, 1],
         [1, 0],
         [0, 1],
         [0, 0]]

    t = [[1, 0], [0, 1], [0, 1], [0, 1]]

    w, epoch = percep_fit(s, t, .2)
    # y = percep_test([-1, 1, -1], w)

    print(w)
    print(epoch)


def test_iris():
    iris = sns.load_dataset('iris')
    # iris = iris.loc[iris['species'] != 'virginica']
    iris = iris.drop(['sepal_width', 'petal_width'], axis=1)
    iris.loc[iris['species'] == 'virginica', 'petal_length'] += 2

    X = iris[['sepal_length', 'petal_length']].to_numpy()
    X = minmax_scale(X)

    y = iris['species'].to_numpy()
    c = {'virginica': [1, -1, -1], 'setosa': [-1, 1, -1], 'versicolor': [-1, -1, 1]}
    y = [c[i] for i in y]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    w, epoch = percep_fit(X_train, y_train)

    # to_class = lambda x: [0 if i == [-1, -1] else 1 if i == [-1, 1] else 2 for i in x]
    # out = list(percep_predict(X_test, w))
    # out = to_class(out)

    # y_test = to_class(y_test)
    # acc = accuracy_score(out, y_test)

    # print(f'Epoch: {epoch}')
    # print(f'Accuracy: {acc}')


if __name__ == '__main__':
    quiz()
