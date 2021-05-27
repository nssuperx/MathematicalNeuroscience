from typing import List
import numpy as np
import matplotlib.pyplot as plt

class amNet:
    result_list = []
    def __init__(self, n: int, m: int, seed: int = 2021):
        # seed
        np.random.seed(seed)
        
        # 定数
        self.N = n
        self.M = m

        # 記憶を作成
        x_value = np.array([-1, 1])
        self.xa = np.random.choice(x_value, (self.M, self.N))

        # 重みの計算
        self.w = np.empty((self.N, self.N))
        for i in range(self.N):
            for j in range(i+1):
                self.w[i][j] = np.sum(self.xa[:, i] * self.xa[:, j])
                self.w[j][i] = self.w[i][j]

        for i in range(self.N):
            self.w[i][i] = 0
        
        self.w = self.w / self.N

        self.result_list = []

    def sgn(self):
        before_x = np.copy(self.x)
        # wb: wight * before_x
        wb = self.w * np.tile(before_x, (self.N, 1))
        wb_sum = np.sum(wb, axis=1)
        return np.where(wb_sum > 0.0, 1.0, -1.0)

    def directon_cos(self):
        return np.sum(self.x * self.xa[self.select_memory, :]) / self.N

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

    def update(self):
        now_result_list = []
        now_result_list.append(self.directon_cos())
        for i in range(self.iteration):
            self.x = self.sgn()
            now_result_list.append(self.directon_cos())
        self.result_list.append(now_result_list)

    def show_result(self):
        for y in self.result_list:
            plt.plot(range(self.iteration + 1), y, linewidth=1.0)
        plt.xlim(-0.1, 12)
        plt.show()
