# create by fanfan on 2018/4/11 0011
import numpy as np
import matplotlib.pyplot  as plt
def loadSimpData():
    datMat = np.matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

def plotData(dataMat,classLabels):
    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []
    for i in range(len(classLabels)):
        if classLabels[i] == 1.0:
            xcord1.append(dataMat[i,0])
            ycord1.append(dataMat[i,1])
        else:
            xcord0.append(dataMat[i,0])
            ycord0.append(dataMat[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord0,ycord0,marker='s',s=90)
    ax.scatter(xcord1,ycord1,marker='o',s=50,c='red')
    plt.title('decision stump test data')
    plt.show()



def stumpClassfify(dataMatrix,dimen,thresVal,threshIenq):
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    if threshIenq == 'lt':
        retArray[dataMatrix[:,dimen] <= thresVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > thresVal] = -1.0
    return  retArray

def buildStump(dataArr,classLabels,D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEnt = np.mat(np.zeros((m,1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1,int(numSteps) + 1):
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassfify(dataMatrix,i,threshVal,inequal)
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                #print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))

                if weightedError < minError:
                    minError = weightedError
                    bestClassEnt = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClassEnt

def adaBoostTrainDS(dataArr,classLabels,numIt = 40):
    """
    基于单层决策树的ada训练
        :param dataArr: 样本特征矩阵
        :param classLabels: 样本分类向量
        :param numIt: 迭代次数
        :return: 一系列弱分类器及其权重,样本分类结果
        """
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels ,D)
        print("D:",D.T)
        alpha = float(0.5 * np.log((1.0 - error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst:",classEst.T)
        print("realClas: ", classLabels)
        expon = np.multiply(-1* alpha * np.mat(classLabels).T,classEst)
        D = np.multiply(D,np.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        print("aggClassEst: ", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))
        errorRate = aggErrors.sum() / m
        print("total error:",errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassfify(dataMatrix,classifierArr[i]['dim'],
                                  classifierArr[i]['thresh'],
                                  classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)

def loadDataSet(fileName):
    numFeat = len(open(fileName,'r',encoding='utf-8').readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def evaluate(aggClassEst,classLabels):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(classLabels)):
        if classLabels[i] == 1.0:
            if (np.sign(aggClassEst[i]) == classLabels[i]):
                TP += 1.0
            else:
                FP += 1.0
        else:
            if (np.sign(aggClassEst[i]) == classLabels[i]):
                TN += 1.0
            else:
                FN += 1.0
    return  TP/(TP + FP),TP/(TP + FN)

def train_test(dataArr,labelArr,dataArrTest,labelArrTest,num):
    classifierArr,aggClassEst = adaBoostTrainDS(dataArr,labelArr,num)
    prTrain = evaluate(aggClassEst,labelArr)
    aggClassEst = adaClassify(dataArrTest,classifierArr)
    prTest = evaluate(aggClassEst,labelArrTest)
    return prTrain,prTest


train_txt = 'horseColicTraining2.txt'
dataMat,classLabels = loadDataSet(train_txt)
test_txt = "horseColicTest2.txt"
dataMat_test,classLabels_test = loadDataSet(test_txt)
prTrain,prTest = train_test(dataMat,classLabels,dataMat_test,classLabels_test,10)
print(prTrain,prTest)