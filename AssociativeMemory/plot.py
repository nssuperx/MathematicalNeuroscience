import matplotlib.pyplot as plt
import csv

data = []
with open('am.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)

for l in data:
    l.pop(-1)

x_index = []
for l in data:
    x_index.append(int(l.pop(0)))

times = len(x_index)
flips = len(data[0])

y_plot = [[0.0] * times for i in range(flips)]
for i in range(flips):
    for j in range(times):
        y_plot[i][j] = float(data[j][i])

for y in y_plot:
    plt.plot(x_index, y)

plt.xlabel("times")
plt.show()