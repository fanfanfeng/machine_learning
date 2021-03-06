# create by fanfan on 2017/8/22 0022
import numpy as np
def loadSimpData():
    datMat = np.matrix([
        [1.,2.1],
        [2.,1.1],
        [1.3,1.],
        [1.,1.],
        [2.,1.]
    ])

    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels

def stumpClassfiy(dataMatrix,dimen,threshVal,threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return  retArray

def buildStump(dataArr,classLabels,D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m,1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1,int(numSteps) + 1):
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassfiy(dataMatrix,i,threshVal,inequal)
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                #print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    return bestStump,minError,bestClassEst

def adaBoostTrainDS(dataArr,classLabels,numIt =40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print("D:",D.T)
        alpha = float(0.5 * np.log((1.0 - error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst: ",classEst.T)
        print("realClas: ",classLabels)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst)
        D = np.multiply(D,np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha * classEst
        print("aggClassEst: ",aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))

        errorRate = aggErrors.sum() /m
        print("total error: ",errorRate,"\n")
        if errorRate == 0.0:
            break
    return weakClassArr

def adaClassify(dataToClass,classifierArr):
    dataMatrix = np.mat(dataToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassfiy(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)

import codecs
def loadDataSet(fileName):
    numFeat = len(codecs.open(fileName,encoding='utf-8').readline().split('\t'))
    dataMat = []
    labeMat = []
    fr  = codecs.open(fileName,encoding='utf-8')
    for line in fr.readlines():
        lineArr = []
        currLine = line.strip().split('\t')
        for l in range(numFeat - 1):
            lineArr.append(float(currLine[-1]))
        dataMat.append(lineArr)
        labeMat.append(float(currLine[-1]))
    return dataMat,labeMat

if __name__ == '__main__':
    dataMat,classLabels = loadSimpData()
    classifierArr = adaBoostTrainDS(dataMat,classLabels,9)
    #print(adaClassify([0,0],classifierArr))
    #print(adaClassify([[5, 5],[0,0]], classifierArr))
    pass






























