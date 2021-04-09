import random
import math
import matplotlib.pyplot as plt
import itertools

# Mirror Symmetry !!!!

# const value
n1 = 6
n2 = 8
mu = 0.8
iterations = 80000

random.seed(2020)

# 1.1 Generate input pairs
input_list = list(itertools.product(map(int, '01'), repeat=6))
D = [[0] * 2 for i in range(len(input_list))]
for i in range(len(input_list)):
    D[i][0] = [1]
    D[i][0].extend(list(input_list[i]))
D_size = len(D)

# 1.2 Generate output pairs
input_size_half = n1 // 2
for d in D:
    cnt = 0
    for i in range(1, input_size_half+1):
        if d[0][i] == d[0][-i]:
            cnt += 1
    if cnt == input_size_half:
        d[1] = 1

class MSNet:
    def __init__(self):
        self.n1 = n1 + 1    # add bias
        self.n2 = n2 + 1

        self.s = [[0.0] * self.n1 for i in range(self.n2)] # s[0]は使用しない
        self.w = [0.0] * self.n2

        self.u = [0.0] * self.n2

        for j in range(1, self.n2):
            for k in range(self.n1):
                self.s[j][k] = random.normalvariate(0,0.1)

        for j in range(self.n2):
            self.w[j] = random.normalvariate(0,0.1)


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
            self.w[j] += mu * r * self.u[j]

        
        rstar = [0.0] * self.n2
        for j in range(1, self.n2):
            rstar[j] = r * self.w[j] * self.u[j] * (1.0 - self.u[j])
            for k in range(self.n1):
                self.s[j][k] += mu * rstar[j] * self.x[k]


    def train(self, iterations=10000, logging=False):
        self.is_logged = logging
        if logging is True:
            self.train_logging(iterations)
            return self.log_E

        self.log_E = [0.0] * iterations
        for i in range(iterations):
            a = random.randint(0, D_size - 1)
            x = D[a][0]
            y = D[a][1]
            self.forward(x, y)
            self.backpropagate()
            self.log_E[i] = self.E

        return self.log_E

    def test(self):
        for d in D:
            x = d[0]
            y = d[1]
            self.forward(x, y)
            print(str(d[0]) + " " + str(d[1]) + " : " + str(self.z))


    def train_logging(self, iterations=10000):
        self.log_E = [0.0] * iterations
        self.log_aE =[[None] * iterations for i in range(D_size)]
        self.log_u = [[0.0] * iterations for i in range(self.n2-1)]
        self.log_w = [[0.0] * iterations for i in range(self.n2)]
        self.log_s = [[[0.0] * iterations for i in range(self.n1)] for j in range(self.n2-1)]
        self.log_z = [0.0] * iterations
        for i in range(iterations):
            a = random.randint(0, D_size - 1)
            x = D[a][0]
            y = D[a][1]
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

    def show_log(self):
        if self.is_logged is False:
            print("Can't show log. Please train(logging=True)")
            return

        x_index = range(iterations)

        fig = plt.figure()
        e = fig.add_subplot(121, xlabel="iterations")
        ae = fig.add_subplot(122, xlabel="iterations")
        e.plot(x_index, self.log_E, linestyle='None', marker=".", ms=2)

        for i in range(D_size):
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
    net = MSNet()
    log_E = net.train(iterations, logging=False)
    net.test()
    
    x_index = range(iterations)
    plt.plot(x_index, log_E, linestyle='None', marker=".", ms=1)
    plt.xlabel("iterations")
    plt.title("error")
    plt.show()

    net.show_log()

    net.print_weight()

if __name__ == "__main__":
    main()