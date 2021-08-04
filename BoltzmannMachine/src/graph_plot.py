import csv
import matplotlib.pyplot as plt

with open('KLdiv.csv') as f:
    reader = csv.reader(f)
    data = [row for row in reader]

x_label = 'itertion'
y_label = 'kl-div'

data_T = [x for x in zip(*data)]

x_list = [int(i) for i in data_T[0]]
y_list = [float(i) for i in data_T[1]]

fig = plt.figure()
ax = fig.add_subplot(111, xlabel=x_label, ylabel=y_label)
# ax.plot(x_list, y_list)
ax.plot(x_list, y_list, linestyle = "None", markersize=3, marker=".")
plt.show()
