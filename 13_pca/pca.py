
# create by fanfan on 2017/9/9 0009
import  numpy as np
def loadDataset(fileName,delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr]
    datArr = [list(map(float,line)) for line in stringArr]
    return np.mat(datArr)

def pca(dataMat,topNfeat = 99999):
    meanVals = np.mean(dataMat,axis=0)
    meanRemoved = dataMat - meanVals
    convMat = np.cov(meanRemoved,rowvar=0)
    eigVals,eigVects = np.linalg.eig(np.mat(convMat))
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat + 1): -1]
    redEigVects = eigVects[:,eigValInd]
    lowDDatamat = meanRemoved * redEigVects
    reconMat = (lowDDatamat * redEigVects.T) + meanVals
    return  lowDDatamat,reconMat

if __name__ == '__main__':
    dataMat = loadDataset('testSet.txt')