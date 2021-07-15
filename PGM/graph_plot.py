import csv
import matplotlib.pyplot as plt

filePathList = ['pgm_sg.csv','pgm_sp.csv','pgm_st.csv']
graphListTypeStr = []

for path in filePathList:
    with open(path) as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    graphListTypeStr.append([x for x in zip(*data)])

graphList = []
for graph in graphListTypeStr:
    graphList.append([[float(i) for i in graph[0]], [float(i) for i in graph[1]]])

x_label = 'FPR'
y_label = 'CDR'

fig = plt.figure()
ax = fig.add_subplot(111, xlabel=x_label, ylabel=y_label)
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
labelList = ['Sg', 'Sp', 'St']
for (graph, labelName) in zip(graphList, labelList):
    ax.plot(graph[0], graph[1], label=labelName)
plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=0.5, fontsize=12)
plt.show()
