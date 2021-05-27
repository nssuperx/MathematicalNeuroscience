from typing import List
import numpy as np
import matplotlib.pyplot as plt

# constant value
n = 1000
m = 80


class amNet:
    result_list = []
    def __init__(self):
        # 記憶を作成
        x_value = np.array([-1, 1])
        self.xa = np.random.choice(x_value, (m, n))

        # 重みの計算
        self.w = np.empty((n, n))
        for i in range(n):
            for j in range(i+1):
                self.w[i][j] = np.sum(self.xa[:, i] * self.xa[:, j])
                self.w[j][i] = self.w[i][j]

        for i in range(n):
            self.w[i][i] = 0
        
        self.w = self.w / n

        self.result_list = []

    def sgn(self):
        before_x = np.copy(self.x)
        # wb: wight * before_x
        wb = self.w * np.tile(before_x, (n, 1))
        wb_sum = np.sum(wb, axis=1)
        return np.where(wb_sum > 0.0, 1.0, -1.0)

    def directon_cos(self):
        return np.sum(self.x * self.xa[self.select_memory, :]) / n

    def flip_set_x(self, a: int):
        self.x = np.copy(self.xa[self.select_memory])
        for i in range(a):
            self.x[i] *= -1

    def run(self, iteration: int, flip_elements_List: List, select_memory: int = 0):
        self.iteration = iteration
        self.select_memory = select_memory
        for flip_num in flip_elements_List:
            print("flip a =  " + str(flip_num))
            self.flip_set_x(flip_num)
            self.update()
        # self.update()

    def update(self):
        now_result_list = []
        now_result_list.append(self.directon_cos())
        for i in range(self.iteration):
            self.x = self.sgn()
            now_result_list.append(self.directon_cos())
        self.result_list.append(now_result_list)

    def show_result(self):
        for y in self.result_list:
            plt.plot(range(self.iteration + 1), y)
        plt.show()

def main():
    net = amNet()
    flip_list = []
    for i in range(0, 500, 25):
        flip_list.append(i)
    net.run(20, flip_list)
    net.show_result()

if __name__ == "__main__":
    main()
