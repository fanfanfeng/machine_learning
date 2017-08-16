# create by fanfan on 2017/8/12 0012
from math import  log
import operator

def calcShannonEnt(dataSet):
    '''
    计算香浓熵
    :param dataSet: 
    :return: 
    '''
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    #香浓商
    shannonEnt = 0.0
    for  key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataset():
    dataSet = [
        [1,1,'yes'],
        [1, 1, 'maybe'],
        [1,0,'no'],
        [0,1,'no'],
        [0,1,'no']
               ]
    labels = ['no surfacing','flippers']
    return dataSet,labels

def splitDataSet(dataSet,axis,value):
    '''
    按照给定特征划分数据集
    :param dataSet: 待划分的数据集
    :param axis: 划分数据集的特征
    :param value: 特征的返回值 
    :return: 
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return  retDataSet

def chooseBestFeatureToSplit(dataSet):
    '''
    选择最好的特征来分类
    :param dataSet: 
    :return: 
    '''
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1

    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]

    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)

    return myTree

def classify(inputTree,featLabels,testVec):
    print(type(inputTree.keys()))
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key :
            if type(secondDict[key]).__name__  == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree,filename):
    '''
    存储决策树
    :param inputTree: 
    :param filename: 
    :return: 
    '''
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return  pickle.load(fr)



if __name__ == '__main__':
    myDat,labels = createDataset()
    print(myDat)
    print(labels)
    print(calcShannonEnt(myDat))

    #print(splitDataSet(myDat,1,1))
    #print(chooseBestFeatureToSplit(myDat))
    import copy
    tree_labels = copy.deepcopy(labels)
    myTree = createTree(myDat,tree_labels)
    print(myTree)
    print(labels)

    class_pridict = classify(myTree,labels,[1,0])
    print(class_pridict)

    storeTree(myTree,'classifierStorage.txt')
    print(grabTree('classifierStorage.txt'))

    import codecs
    with codecs.open('lenses.txt',encoding='utf-8') as fr:
        lenses = [lnst.strip().split('\t') for lnst in fr.readlines()]
        lensesLabels = ['age','prescript','astigmatic','tearRate']
        lensesTree = createTree(lensesLabels,lenses)
        print(lensesTree)