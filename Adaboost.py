# coding:utf-8
"""
作者：zhaoxingfeng	日期：2017.04.19
功能：Adaboost，分类，马疝气病数据集
版本：V2.0
参考文献：
[1] 百度技术.Boosting算法简介[DB/OL].http://baidutech.blog.51cto.com/4114344/743809/,2011-01-18.
[2] 李闯，丁晓青，吴佑寿. 一种改进的AdaBoost算法——AD AdaBoost[J].计算机学报，2007，30(1)：353-368
"""
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 记录运行时间
def runTime(func):
    def wrapper(*args, **kwargs):
        import time
        t1 = time.time()
        func(*args, **kwargs)
        t2 = time.time()
        print("{0} runTime: {1:.2f}s".format(func.__name__, t2 - t1))
    return wrapper

# 单层决策树
def singleTree(dataArr, dim, threshVal, threshFlag):
    returnList = []
    for i in dataArr[:, dim]:
        if threshFlag == "low":
            if i <= threshVal:
                returnList.append(-1)
            else:
                returnList.append(1)
        else:
            if i >= threshVal:
                returnList.append(-1)
            else:
                returnList.append(1)
    return np.array(returnList)

# 建树
def buildTree(dataArr, labelArr, weight):
    m, n = dataArr.shape
    bestTree = {}
    minErro = np.inf
    numSteps = 10
    for i in range(n):
        # 分类阈值，固定步进值
        min_i, max_i = min(dataArr[:, i]), max(dataArr[:, i])
        stepSize = (max_i - min_i) / numSteps
        threshVals = [min_i+j*stepSize for j in range(-1, numSteps+1)]
        """
        # 对所有数据按照从大到小排序后两两求均值，作为分类阈值，该维度数据取值多时会比较慢，但准确率比固定步数要高2%
        sortedDim = sorted(list(set(dataArr[:, i])))
        threshVals = [(sortedDim[j] + sortedDim[j+1]) / 2 for j in range(len(sortedDim)-1)]
        threshVals.insert(0, sortedDim[0] / 2)
        threshVals.append(sortedDim[-1] * 1.5)
        """
        for threshVal in threshVals:
            for flag in ["low", "upper"]:
                labelPre = singleTree(dataArr, i, threshVal, flag)
                erroList = [0 if x == y else 1 for x, y in zip(labelArr, labelPre)]
                # 计算加权样本误差总和
                weightErro = sum(weight * np.array(erroList))
                if weightErro < minErro:
                    minErro = weightErro
                    bestTree['dim'] = i
                    bestTree['thresh'] = threshVal
                    bestTree['flag'] = flag
                    bestlabelPre = np.copy(labelPre)
    return bestTree, minErro, bestlabelPre

# 主程序
def adaboostTrain(dataArr, labelArr, maxIter=50):
    weakModel = []
    m, n = dataArr.shape
    weight = np.ones(m) / m
    addLabelPre = np.zeros(m)
    accuracyList = []
    for i in range(maxIter):
        print("iter = {}".format(i))
        bestTree, erro, labelPre = buildTree(dataArr, labelArr, weight)
        # 避免除0错误
        alpha = 0.5 * np.log((1 - erro) / max(erro, 1e-15))
        bestTree['alpha'] = alpha
        weakModel.append(bestTree)
        # 更新每个样本的权值weight
        for j in range(m):
            if labelPre[j] == labelArr[j]:
                weight[j] *= np.exp(-1 * alpha)
            else:
                weight[j] *= np.exp(+1 * alpha)
        weight /= sum(weight)
        addLabelPre += alpha * labelPre
        addErro = [1 if x == np.sign(y) else 0 for x, y in zip(labelArr, addLabelPre)]
        accuracy = sum(addErro) / m
        accuracyList.append(accuracy)
        # 全部分类正确则提前退出循环
        if accuracy == 1:
            break
    return weakModel, addLabelPre, accuracyList

# 对测试样本进行分类
def adaboostClassify(classifyArr, testArr):
    addLabelPre = np.zeros(testArr.shape[0])
    for i in range(len(classifyArr)):
        labelPre = singleTree(testArr, classifyArr[i]['dim'],
                              classifyArr[i]['thresh'], classifyArr[i]['flag'])
        addLabelPre += classifyArr[i]['alpha'] * labelPre
    return np.sign(addLabelPre)

# 迭代误差图
def plotAccuracy(lst):
    x = range(len(lst))
    plt.plot(x, lst)
    plt.xlabel('Iter')
    plt.ylabel('accuracy')
    plt.show()

@runTime
def test():
    # 训练样本
    train = pd.read_csv('horseColicTraining.txt', header=None, sep='\t')
    train[21] = train[21].map({0: -1, 1: 1})
    train = train.values
    trainArr, labelArr = train[:, :-1],  train[:, -1]
    weakModel, addLabel, accuracyList = adaboostTrain(trainArr, labelArr, 30)
    print("accuracy in train = {}%".format(accuracyList[-1] * 100))
    print("model = {}".format(weakModel))
    plotAccuracy(accuracyList)
    # 测试样本
    testData = pd.read_csv('horseColicTest.txt', header=None, sep='\t')
    testData[21] = testData[21].map({0: -1, 1: 1})
    testData = testData.values
    predict = adaboostClassify(weakModel, testData)
    # 统计分类正确率
    accuracy = sum([1 if x[0] == x[1] else 0 for x in zip(testData[:, -1], predict)]) / len(testData)
    print("accuracy in testData = {}%".format(accuracy * 100))
test()
