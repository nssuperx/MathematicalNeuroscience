import random
import math
import matplotlib.pyplot as plt

# XOR Problem !!!

random.seed(2021)

class Net:
    def __init__(self, data, n1, n2, mu, iterations, detail_logging=False):
        # 使うデータ，入力と期待する出力のペア
        self.D = data
        self.D_size = len(self.D)
        # 学習係数
        self.mu = mu
        # 繰り返し回数
        self.iterations = iterations
        # ログをたくさん取るかのフラグ
        self.is_detail_logging = detail_logging

        # 入力数と中間層(2層目)の数を設定
        self.n1 = n1 + 1    # add bias
        self.n2 = n2 + 1

        # 重み
        self.s = [[0.0] * self.n1 for i in range(self.n2)] # s[0]は使用しない
        self.w = [0.0] * self.n2

        # 中間層の出力
        self.u = [0.0] * self.n2

        # 出力
        self.z = 0.0

        # 誤差
        self.E = 0.0
        # 誤差のログ
        self.log_E = [0.0] * self.iterations

        # 正解率のログ
        self.log_correct_rate = []

        # ニューラルネットに与える入力
        self.x = None
        # 期待する出力
        self.y = None

        # 初期値代入（ランダム）
        for j in range(1, self.n2):
            for k in range(self.n1):
                self.s[j][k] = random.normalvariate(0,0.1)

        for j in range(self.n2):
            self.w[j] = random.normalvariate(0,0.1)

        # 詳細ログ取得用変数宣言
        if(detail_logging):
            self.log_aE =[[None] * self.iterations for i in range(self.D_size)]
            self.log_u = [[0.0] * self.iterations for i in range(self.n2-1)]
            self.log_w = [[0.0] * self.iterations for i in range(self.n2)]
            self.log_s = [[[0.0] * self.iterations for i in range(self.n1)] for j in range(self.n2-1)]
            self.log_z = [0.0] * self.iterations

        # 学習後の重み
        '''
        self.s[1][0] = -2.6906800130974426
        self.s[1][1] = 6.393982015907248  
        self.s[1][2] = 6.415342116315495  
        self.s[2][0] = -6.1787881128386735
        self.s[2][1] = 4.020462432141655  
        self.s[2][2] = 4.0461390770042005
        self.w[0] = -3.7685246774464694
        self.w[1] = 8.304037390324421
        self.w[2] = -8.94283529977682
        '''


    def forward(self, x, y):
        self.x = x
        self.y = y

        self.u[0] = 1.0

        for j in range(1, self.n2):
            sum = 0.0
            for k in range(self.n1):
                sum += self.s[j][k] * self.x[k]
            self.u[j] = sigmoid(sum)

        sum = 0.0
        for j in range(self.n2):
            sum += self.w[j] * self.u[j]
        self.z = sigmoid(sum)
        self.E = (self.z - self.y) * (self.z - self.y) * 0.5


    def backpropagate(self):
        r = (self.y - self.z) * self.z * (1.0 - self.z)
        for j in range(self.n2):
            self.w[j] += self.mu * r * self.u[j]
        
        rstar = [0.0] * self.n2
        for j in range(1, self.n2):
            rstar[j] = r * self.w[j] * self.u[j] * (1.0 - self.u[j])
            for k in range(self.n1):
                self.s[j][k] += self.mu * rstar[j] * self.x[k]


    def train(self, iterations=10000, progress_interval=10):
        for i in range(iterations):
            a = random.randint(0, self.D_size - 1)
            x = self.D[a][0]
            y = self.D[a][1]
            self.forward(x, y)
            self.backpropagate()
            self.log_E[i] = self.E
            if(i % progress_interval == 0):
                self.calc_and_logging_correct_rate()

        return self.log_E


    def train_with_detail_logging(self, iterations=10000):
        if self.is_detail_logging is False:
            print("Can't logging. Please (detail_logging=False) when class initialize.")
            return
        
        for i in range(iterations):
            a = random.randint(0, self.D_size - 1)
            x = self.D[a][0]
            y = self.D[a][1]
            self.forward(x, y)
            self.backpropagate()
            self.log_E[i] = self.E
            self.log_aE[a][i] = self.E

            for j in range(self.n2-1):
                for k in range(self.n1):
                    self.log_s[j][k][i] = self.s[j+1][k]

            for j in range(self.n2-1):
                self.log_u[j][i] = self.u[j+1]

            for j in range(self.n2):
                self.log_w[j][i] = self.w[j]

            self.log_z[i] = self.z

        return self.log_E

    def calc_and_logging_correct_rate(self):
        sum_correct = 0.0
        for d in self.D:
            x = d[0]
            y = d[1]
            self.forward(x, y)
            if(y == 0):
                sum_correct += 1 - self.z
            else:
                sum_correct += self.z
        self.log_correct_rate.append(sum_correct/self.D_size)

    def test(self):
        for d in self.D:
            x = d[0]
            y = d[1]
            self.forward(x, y)
            print(str(d[0]) + " " + str(d[1]) + " : " + str(self.z))


    def show_log(self):
        if self.is_detail_logging is False:
            print("Can't show log. Please (detail_logging=False) when class initialize.")
            return

        x_index = range(self.iterations)

        fig = plt.figure()
        e = fig.add_subplot(121, xlabel="iterations")
        ae = fig.add_subplot(122, xlabel="iterations")
        e.plot(x_index, self.log_E, linestyle='None', marker=".", ms=2)

        for i in range(self.D_size):
            ae.plot(x_index, self.log_aE[i], linestyle='None', marker=".", ms=2)

        fig.suptitle("error")
        plt.show()

        fig = plt.figure()
        s = fig.add_subplot(221, xlabel="iterations", title="weight s")
        u = fig.add_subplot(222, xlabel="iterations", title="hidden u")
        w = fig.add_subplot(223, xlabel="iterations", title="weight w")
        z = fig.add_subplot(224, xlabel="iterations", title="out z")

        for j in range(self.n2-1):
            for k in range(self.n1):
                s.plot(x_index, self.log_s[j][k], linestyle='None', marker=".", ms=1)

        for j in range(self.n2-1):
            u.plot(x_index, self.log_u[j], linestyle='None', marker=".", ms=1)

        for j in range(self.n2):
            w.plot(x_index, self.log_w[j], linestyle='None', marker=".", ms=1)

        z.plot(x_index, self.log_z, linestyle='None', marker=".", ms=1)
        
        plt.show()

    def print_weight(self):
        for j in range(1, self.n2):
            for k in range(self.n1):
                print("s[" + str(j) + "][" + str(k) + "]" " = " + str(self.s[j][k]))

        for j in range(self.n2):
            print("w[" + str(j) + "]" + " = " + str(self.w[j]))

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def main():
    # constant value
    D = [[[1,0,0],0],[[1,0,1],1],[[1,1,0],1],[[1,1,1],0]]

    n1 = 2
    n2 = 2
    mu = 0.7
    iterations = 10000
    progress_interval = 10

    xorNet = Net(D, n1, n2, mu, iterations)
    log_E = xorNet.train(iterations, progress_interval)
    xorNet.test()
    
    x_index = range(iterations)
    plt.plot(x_index, log_E, linestyle='None', marker=".", ms=1)
    plt.xlabel("iterations")
    plt.title("error")
    plt.show()

    # 正解率
    x_index = range(0, iterations, progress_interval)
    plt.plot(x_index, xorNet.log_correct_rate, linestyle='None', marker=".", ms=1)
    plt.xlabel("iterations")
    plt.title("percentage of correct responses")
    plt.show()

    xorNet.show_log()

    xorNet.print_weight()

if __name__ == "__main__":
    main()