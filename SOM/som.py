import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# constant value
# N = 256
NX = 8
NY = 8
DIM = 2     #dimension
iterations = 100000
alpha = 0.2
sigma = 0.8
sigma2 = 2 * sigma**2
exp_table = {}

# NOTE: p(x)は2次元, xとy

class som:
    def __init__(self):
        '''
        self.m = [[[0.0] * DIM for i in range(NX)] for j in range(NY)]
        for i in range(NY):
            for j in range(NX):
                for k in range(DIM):
                    self.m[i][j][k] = random.uniform(-0.1, 0.1)
        '''
        self.m = np.random.default_rng().uniform(-0.5, 0.5, size=(NY, NX, DIM))
        # print(self.m.shape)

    def activity_dynamics(self, x, y):
        # min_distance = float('inf')
        distance = (x - self.m[:,:,0])**2 + (y - self.m[:,:,1])**2
        self.rc_x, self.rc_y = np.unravel_index(np.argmin(distance), distance.shape)
        '''
        for i in range(NY):
            for j in range(NX):
                distance = (x - self.m[i][j][0])**2 + (y - self.m[i][j][1])**2
                if(min_distance > distance):
                    min_distance = distance
                    self.rc_x = i
                    self.rc_y = j
        '''

    def learning_dynamics(self, x, y):
        for i in range(NY):
            for j in range(NX):
                distance = (self.rc_x - i)**2 + (self.rc_y - j)**2
                self.m[i][j][0] += alpha * (x - self.m[i][j][0]) * exp_table[distance]
                self.m[i][j][1] += alpha * (y - self.m[i][j][1]) * exp_table[distance]


    def update(self):
        plts = []
        fig = plt.figure()

        for i in range(iterations + 1):
            lines = []
            
            p0x = 0
            p0y = 1
            p1x = -1
            p1y = -1
            p2x = 1
            p2y = -1
            Area = 0.5 *(-p1y*p2x + p0y*(-p1x + p2x) + p0x*(p1y - p2y) + p1x*p2y)
            x = 0
            y = 0
            while True:
                x = random.uniform(-1, 1)
                y = random.uniform(-1, 1)
                s = 1/(2*Area)*(p0y*p2x - p0x*p2y + (p2y - p0y)*x + (p0x - p2x)*y)
                t = 1/(2*Area)*(p0x*p1y - p0y*p1x + (p0y - p1y)*x + (p1x - p0x)*y)
                if ((0 < s < 1) and (0 < t < 1) and (0<1-s-t<1)):
                    break

            ''' rect
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            '''
            self.activity_dynamics(x, y)
            self.learning_dynamics(x, y)

            if i % 1000 == 0:
                print('i = ' + str(i))
                # 横
                for j in range(NY):
                    for k in range(1, NX):
                        im = plt.plot([self.m[j][k-1][0], self.m[j][k][0]], [self.m[j][k-1][1], self.m[j][k][1]], color='red')
                        lines.extend(im)

                # 縦
                for j in range(1, NY):
                    for k in range(NX):
                        im = plt.plot([self.m[j-1][k][0], self.m[j][k][0]], [self.m[j-1][k][1], self.m[j][k][1]], color='red')
                        lines.extend(im)
                        
                lines.append(plt.text(0.0, 1.1, ("iteration = " + str(i)), ha="center", va="bottom", fontsize="large"))
                plts.append(lines)

        anim = animation.ArtistAnimation(fig, plts, interval=200, repeat_delay=500)
        anim.save('./figure/n' + str(NX) + 'x' + str(NY) + '_2d_triangle.gif', writer="imagemagick")
        plt.show()

def make_exp_table():
    global exp_table
    for i in range(NY):
        for j in range(NX):
            distance = i**2 + j**2
            exp_table[distance] = math.exp(-distance / sigma2)


def main():
    make_exp_table()
    net = som()
    net.update()

if __name__ == "__main__":
    main()
