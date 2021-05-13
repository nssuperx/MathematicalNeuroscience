import random
import math
import matplotlib.pyplot as plt
import itertools

from mynet import Net

# Mirror Symmetry !!!!

random.seed(2021)

def main():
    # const value
    n1 = 6
    n2 = 5
    # n2_list = [1, 5, 10, 25, 50, 100]
    mu = 0.8
    iterations = 100000
    progress_interval = 10

    # データ作成
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
    
    # ここから実験
    # for n2 in n2_list:
    MSNet = Net(D, n1, n2, mu, iterations, detail_logging=True)
    # log_E = MSNet.train(progress_interval)
    log_E = MSNet.train_with_detail_logging(progress_interval)
    MSNet.test()
    
    x_index = range(iterations)
    plt.plot(x_index, log_E, linestyle='None', marker=".", ms=1)
    plt.xlabel("iterations")
    plt.title("error")
    plt.show()

    # 正解率
    x_index = range(0, iterations, progress_interval)
    plt.plot(x_index, MSNet.log_correct_rate, linestyle='None', marker=".", ms=1)
    plt.xlabel("iterations")
    plt.title("percentage of correct responses")
    plt.show()

    MSNet.show_log()

    MSNet.print_weight()

if __name__ == "__main__":
    main()
