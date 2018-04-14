# create by fanfan on 2018/4/8 0008
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr:
        lineArr = fr.readline().strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def plotBestFit(weights):
    dataMat,labeMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labeMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    if weights is not None:
        x = np.arange(-3.0,3.0,0.1)
        y = (-weights[0] - weights[1] *x) /weights[2]
        ax.plot(x,y)
    plt.xlabel("x1")
    plt.ylabel('x2')
    plt.show()

def sigmoid(inX):
    return 1.0/(1 + np.exp(-inX))

def gradAscent(dataMatIn,classLabels,history_weight):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h )
        weights += alpha * dataMatrix.transpose() * error
        history_weight.append(np.copy(weights))
    return weights

history_weight = []
dataMat,labelMat = loadDataSet()
gradAscent(dataMat,labelMat,history_weight)
fig = plt.figure()
currentAxis = plt.gca()
ax = fig.add_subplot(111)
line, = ax.plot([],[],'b',lw=2)

def draw_line(weights):
    x = np.arange(-5.0,5.0,0.1)
    y = (-weights[0] - weights[1] *x)/weights[2]
    line.set_data(x,y)
    return line,

def init():
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    plt.xlabel("x1")
    plt.ylabel('x2')
    plt.show()

def animate(i):
    return draw_line(history_weight[i])

anim = animation.FuncAnimation(fig,animate,init_func=init,
                               frames=len(history_weight),
                               blit=True)
plt.show()
anim.save('gradAscent.gif',fps=2,writer='imagemagick')


