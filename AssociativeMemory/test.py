from am import amNet

# constant value
n = 1000
m = 200

def main():
    net = amNet(n, m)
    flip_list = []
    for i in range(0, 500, 25):
        flip_list.append(i)
    net.run(100, flip_list)
    net.show_result()
    # net.show_image()

if __name__ == "__main__":
    main()
