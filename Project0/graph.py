import csv
import matplotlib.pyplot as plt

with open('project0_2.csv') as f:
    reader = csv.reader(f)
    data = [row for row in reader]

x_label = 't'
y_label = 'x(t)'

data_T = [x for x in zip(*data)]

print(len(data_T))
# csvで書き出したときに，最後の","をつけているので，それを消す．
data_T.pop()

x_list = range(51)
y_lists = []

for data_row in data_T:
    y_lists.append([float(i) for i in data_row])

fig = plt.figure()
ax = fig.add_subplot(111, xlabel=x_label, ylabel=y_label)
for y_list in y_lists:
    ax.plot(x_list, y_list)
plt.show()
