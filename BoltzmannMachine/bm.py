import numpy as np

class BoltzmannMachine:
    def __init__(self, neuron: int=3) -> None:
        self.x = self.set_default_state(neuron)
        self.w = self.set_default_weight()

    def set_default_state(self, neuron: int=3) -> np.ndarray:
        x = np.zeros((neuron+1))
        x[0] = 1
        return x.copy()

    def set_default_weight(self) -> np.ndarray:
        return np.zeros((self.x.shape[0], self.x.shape[0]))

    def change_state(self, neuron_number: int) -> int:
        T = 1
        u = np.dot(self.w[neuron_number, :], self.x)
        prob = 1 / (1 + np.exp(-u / T))
        self.x[neuron_number] = 1 if prob < np.random.rand() else 0
        return self.x[neuron_number]

    def calc_stationary_dist(self):
        '''
        numpyの機能を使いたいので，本来の計算方法とは異なる．
        '''
        w_triu = np.triu(self.w)
        x_tmp = np.tile(self.x, (self.x.shape[0], 1)) * np.tile(self.x, (self.x.shape[0], 1)).T
        E = -np.sum(np.multiply(w_triu, x_tmp))
        pass
