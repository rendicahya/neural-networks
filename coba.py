import numpy as np

p = np.array([[0, 0], [0, 1], [1, 0]])
r = [[0.0600547, 0.75866121],
     [0.95049736, 0.17046602],
     [0.00596853, 0.15488992]]

d = [np.argmin(np.sum(abs(i - p), axis=1)) for i in r]

print(d)
