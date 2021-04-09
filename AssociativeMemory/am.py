import math
import random
import matplotlib.pyplot as plt

# constant value
n = 1000
m = 80


class amNet:
    def __init__(self):
        self.x = [[1] * n for i in range(m)]

        for a in range(m):
            for i in range(n):
                if random.randint(0, 1) == 0:
                    self.x[a][i] = -1
        
        self.w = [[0.0] * n for i in range(n)]

        for i in range(n):
            for j in range(n):
                sum = 0.0
                for a in range(m):
                    sum += self.x[a][i] * self.x[a][j]
                self.w[i][j] = sum / n

                



def main():
    net = amNet()

if __name__ == "__main__":
    main()