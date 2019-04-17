import numpy as np

p = np.array([[0, 0], [0, 1], [1, 0]])
r = [0.0600547, 0.75866121]
d = np.argmin([sum(abs(i - r)) for i in p])

print(d)
