# create by fanfan on 2017/9/9 0009
import numpy as np
import random
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curline = line.strip().split('\t')
        fltLine = list(map(float,curline))
        dataMat.append(fltLine)
    return dataMat

def disEclud(vecA,vecB):
    return np.sqrt(sum(pow(vecA - vecB,2)))

def randCent(dataSet,k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:j] =  np.mat(minJ + rangeJ * np.random.rand(k,1))

    return centroids

def kMeans(dataSet,k,distMeans=disEclud,createCent = randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    centroids = createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeans(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist ** 2

        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:.0].A == cent)][0]
            centroids[cent,:] = np.mean(ptsInClust,axis=0)

    return centroids,clusterAssment

if __name__ == '__main__':
    dataMat = np.mat(loadDataSet("testSet.txt"))
    myCentroids,clustAssing = kMeans(dataMat,4)