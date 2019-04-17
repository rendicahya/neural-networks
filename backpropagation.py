import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import accuracy_score


def sig(X):
   return [1 / (1 + np.exp(-x)) for x in X]


def sigd(X):
   for i, x in enumerate(X):
      s = sig([x])[0]

      yield s * (1 - s)


def bp_fit(C, X, t, a, mep, mer):
   # Inisialisasi
   # nin: neuron input
   nin = [np.empty(i) for i in C]

   # n: neuron
   n = [np.empty(j + 1) if i < len(C) - 1 else np.empty(j) for i, j in enumerate(C)]

   # w: weight
   w = np.array([np.random.rand(C[i] + 1, C[i + 1]) for i in range(len(C) - 1)])

   # dw: delta weight
   dw = [np.empty((C[i] + 1, C[i + 1])) for i in range(len(C) - 1)]

   # d: delta
   d = [np.empty(s) for s in C[1:]]

   # din: delta input
   din = [np.empty(s) for s in C[1:-1]]

   # ep: epoch
   ep = 0

   # mse: mean square error
   mse = 1

   #
   for i in range(0, len(n) - 1):
      n[i][-1] = 1

   while ep < mep and mse > mer:
      ep += 1
      mse = 0

      for r in range(len(X)):
         n[0][:-1] = X[r]

         for L in range(1, len(C)):
            nin[L] = np.dot(n[L - 1], w[L - 1])
            n[L][:len(nin[L])] = sig(nin[L])

         e = t[r] - n[-1]
         mse += sum(e ** 2)
         d[-1] = e * list(sigd(nin[-1]))
         dw[-1] = a * d[-1] * n[-2].reshape((-1, 1))

         for L in range(len(C) - 1, 1, -1):
            din[L - 2] = np.dot(d[L - 1], np.transpose(w[L - 1][:-1]))
            d[L - 2] = din[L - 2] * np.array(list(sigd(nin[L - 1])))
            dw[L - 2] = (a * d[L - 2]) * n[L - 2].reshape((-1, 1))

         w += dw

      mse /= len(X)

   return w, ep, mse


def bp_predict(X, w):
   n = [np.empty(len(i)) for i in w]
   nin = [np.empty(len(i[0])) for i in w]

   n.append(np.empty(len(w[-1][0])))

   for x in X:
      n[0][:-1] = x

      for L in range(0, len(w)):
         nin[L] = np.dot(n[L], w[L])
         n[L + 1][:len(nin[L])] = sig(nin[L])

      yield n[-1].copy()


def test3layer():
   c = 3, 2, 2
   X = [[.8, .2, .1],
        [.1, .8, .9]]
   y = [[0, 0],
        [0, 1]]
   w = np.array([[[.1, .2],
                  [.3, .4],
                  [.5, .6],
                  [.7, .8]],
                 [[.1, .4],
                  [.2, .5],
                  [.3, .6]]])

   w, ep, mse = bp_fit(c, X, y, .1, 100, .1, w)
   y = bp_predict([.8, .2, .1], w)

   print(y)


def test5layer():
   pass
   # w = np.array([[[.1, .2],
   #                [.3, .4],
   #                [.5, .6],
   #                [.7, .8]],
   #               [[.1, .4],
   #                [.2, .5],
   #                [.3, .6]],
   #               [[.3, .4, .5],
   #                [.6, .5, .4],
   #                [.8, .7, .6]]])


def testiris():
   c = 4, 3, 2

   iris = load_iris()
   X = minmax_scale(iris.data)

   y = iris.target
   p = np.array([[0, 0], [0, 1], [1, 0]])

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
   w, ep, mse = bp_fit(c, X_train, [p[i] for i in y_train], .1, 1000, .1)

   print('Epoch: %d' % ep)
   print('MSE: %f' % mse)

   o = list(bp_predict(X_test, w))
   oc = [np.argmin(np.sum(abs(i - p), axis=1)) for i in o]
   acc = accuracy_score(oc, y_test)

   print('Output: {}'.format(oc))
   print('True  : {}'.format(y_test))
   print('Accuracy: %f' % acc)


if __name__ == '__main__':
   testiris()
