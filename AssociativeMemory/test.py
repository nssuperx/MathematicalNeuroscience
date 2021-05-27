from am import amNet

# constant value
n = 1000
m = 80

def main():
    net = amNet(n, m)
    flip_list = []
    for i in range(0, 500, 25):
        flip_list.append(i)
    net.run(20, flip_list)
    net.show_result()

if __name__ == "__main__":
    main()
