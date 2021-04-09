import numpy as np
import matplotlib.pyplot as plt

T = np.array([[6,-3,-7],[-1,2,1],[5,-3,-6]])
print(T)

x = np.empty((51,3))
x[0][0], x[0][1], x[0][2] = (4,0,3)

for i in range(1,51):
    Tx = T.dot(x[i-1])
    x[i] = Tx / np.linalg.norm(Tx)

x_index = np.array(range(51))

for i in range(0, 51, 5):
    print("index:" + str(i) + str(x[i]))

x = x.T
plt.plot(x_index, x[0])
plt.plot(x_index, x[1])
plt.plot(x_index, x[2])
plt.show()
