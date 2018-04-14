# create by fanfan on 2018/3/26 0026
import operator
from math import log
def majorityCnt(classList):
    """
    返回出现次数最多的分类名称
        :param classList: 类列表
        :return: 出现次数最多的类名称
        """
    classCount = {}
    for vote in classList:
        classCount.setdefault(vote,0)
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def chooseBestFeatureToSplitByID3(dataSet):
    """
    选择最好的数据集划分方式
        :param dataSet:
        :return:
        """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeatrue = -1
    for i in range(numFeatures):
        infoGain = calcInformationGain(dataSet,baseEntropy,i)
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeatrue = i
    return bestFeatrue

def createTree(dataSet,labels,chooseBestFeatureToSplitFunc=chooseBestFeatureToSplitByID3):
    """
    创建决策树
        :param dataSet:数据集
        :param labels:数据集每一维的名称
        :return:决策树
        """
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplitFunc(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = { bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for  example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


def createDataSet():
    """
创建数据集

    :return:
    """
    dataSet = [[u'青年', u'否', u'否', u'一般', u'拒绝'],
               [u'青年', u'否', u'否', u'好', u'拒绝'],
               [u'青年', u'是', u'否', u'好', u'同意'],
               [u'青年', u'是', u'是', u'一般', u'同意'],
               [u'青年', u'否', u'否', u'一般', u'拒绝'],
               [u'中年', u'否', u'否', u'一般', u'拒绝'],
               [u'中年', u'否', u'否', u'好', u'拒绝'],
               [u'中年', u'是', u'是', u'好', u'同意'],
               [u'中年', u'否', u'是', u'非常好', u'同意'],
               [u'中年', u'否', u'是', u'非常好', u'同意'],
               [u'老年', u'否', u'是', u'非常好', u'同意'],
               [u'老年', u'否', u'是', u'好', u'同意'],
               [u'老年', u'是', u'否', u'好', u'同意'],
               [u'老年', u'是', u'否', u'非常好', u'同意'],
               [u'老年', u'否', u'否', u'一般', u'拒绝'],
               ]
    labels = [u'年龄', u'有工作', u'有房子', u'信贷情况']
    # 返回数据集和每个维度的名称
    return dataSet, labels

def splitDataSet(dataSet,axis,value):
    """
    按照给定特征划分数据集
        :param dataSet: 待划分的数据集
        :param axis: 划分数据集的特征的维度
        :param value: 特征的值
        :return: 符合该特征的所有实例（并且自动移除掉这维特征）
        """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)

    return retDataSet

def calcShannonEnt(dataSet):
    """
    计算训练数据集中的Y随机变量的香农熵
        :param dataSet:
        :return:
        """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = labelCounts[key]/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def calcConditionalEntropy(dataSet,i,featList,uniqueVals):
    '''
        计算X_i给定的条件下，Y的条件熵
        :param dataSet:数据集
        :param i:维度i
        :param featList: 数据集特征列表
        :param uniqueVals: 数据集特征集合
        :return:条件熵
        '''
    ce = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet,i,value)
        prob = len(subDataSet) / len(dataSet)
        ce += prob * calcShannonEnt(subDataSet)
    return ce

def calcInformationGain(dataSet,baseEntropy,i):
    """
        计算信息增益
        :param dataSet:数据集
        :param baseEntropy:数据集中Y的信息熵
        :param i: 特征维度i
        :return: 特征i对数据集的信息增益g(dataSet|X_i)
        """
    featList = [example[i] for example in dataSet]
    uniqueVals = set(featList)
    newEntropy = calcConditionalEntropy(dataSet,i,featList,uniqueVals)
    infoGain = baseEntropy - newEntropy
    return infoGain

def calcInfomationGainRate(dataSet,baseEntropy,i):
    """
        计算信息增益比
        :param dataSet:数据集
        :param baseEntropy:数据集中Y的信息熵
        :param i: 特征维度i
        :return: 特征i对数据集的信息增益g(dataSet|X_i)
        """
    return calcInformationGain(dataSet,baseEntropy,i)/baseEntropy



myDat,labels = createDataSet()
myTree = createTree(myDat,labels)

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
import sys
import os
sys.path.append( os.path.dirname(__file__))
import treePlotter
treePlotter.createPlot(myTree)





