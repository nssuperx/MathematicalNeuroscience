import random
import math
import matplotlib.pyplot as plt

from mynet import Net

# XOR Problem !!!

random.seed(2021)

def main():
    # constant value
    D = [[[1,0,0],0],[[1,0,1],1],[[1,1,0],1],[[1,1,1],0]]

    n1 = 2
    n2 = 2
    # n2_list = [1, 5, 10, 25, 50, 100]
    mu = 0.7
    iterations = 10000
    progress_interval = 10

    # for n2 in n2_list:
    xorNet = Net(D, n1, n2, mu, iterations, detail_logging=True)
    log_E = xorNet.train_with_detail_logging(progress_interval)
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
