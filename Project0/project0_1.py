import numpy as np
import matplotlib.pyplot as plt

n = 100

A = np.empty((n,n))
x = np.empty((n))
y = np.empty((n))

for i in range(n):
    for j in range(n):
        A[i][j] = i + j + 1

for i in range(n):
    x[i] = i

for i in range(n):
    y[i] = A[i].dot(x)

print(y[0])
print(y[n-1])

x_index = range(100)
plt.plot(x_index, y)
plt.show()