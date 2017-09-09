# create by fanfan on 2017/9/7 0007
import numpy as np
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        currLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(currLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(currLine[-1]))
    return dataMat,labelMat

def standRegress(xArr,yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T* xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular,cannot do inverse")
    ws = xTx.I * (xMat.T * yMat)
    return ws

if __name__ == '__main__':
    xArr,yArr = loadDataSet('ex0.txt')
    ws = standRegress(xArr,yArr)
    print(ws)
