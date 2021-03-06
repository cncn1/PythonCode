#!/usr/bin/python
# -*- coding:utf-8 -*-
from math import log
from numpy import array
from FSFOATOOL import *


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:  # the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt


# 根据信息熵计算最优特征,代码引用机器学习实战决策树一章
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        if infoGain > bestInfoGain:  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer


# 根据计算信息增益比计算最优特征
def chooseBestFeatureByRatioToSplit(dataSet):
    """
    输入：数据集
    输出：最好的划分维度
    描述：选择最好的数据集划分维度
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGainRatio = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        splitInfo = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            splitInfo += -prob * log(prob, 2)
        infoGain = baseEntropy - newEntropy
        if (splitInfo == 0):  # fix the overflow bug
            continue
        infoGainRatio = infoGain / splitInfo
        if infoGainRatio > bestInfoGainRatio:
            bestInfoGainRatio = infoGainRatio
            bestFeature = i
    return bestFeature


# 做最后阶段的群体选优策略
def GroupSelection(optimal_area, feature_total, generate_num):
    count_each_feature = [0] * feature_total  # 根据每棵树的特征统计最优森林中所有特征总共出现的次数
    last_compare_subset_accuracy = [0] * feature_total
    last_compare_subset_DR = [0] * feature_total
    for optimal_each_tree in optimal_area:
        for feature_index in xrange(feature_total):
            if optimal_each_tree.list[feature_index] == 1:
                count_each_feature[feature_index] += 1
    optimal_feature_index = array(count_each_feature).argsort()  # 每个特征出现的总次数降序排列后的下标
    # 用如下这种结构的写法整好可以保证last_compare_subset_DR中的特征比last_compare_subset_accuracy中的特征少1
    index = feature_total
    while generate_num > 1:
        index -= 1
        last_compare_subset_DR[optimal_feature_index[index]] = 1
        last_compare_subset_accuracy[optimal_feature_index[index]] = 1
        generate_num -= 1
    last_compare_subset_accuracy[optimal_feature_index[index - 1]] = 1
    last_compare_subset_accuracy = listIntToStr(last_compare_subset_accuracy)
    last_compare_subset_DR = listIntToStr(last_compare_subset_DR)
    return last_compare_subset_accuracy, last_compare_subset_DR


def OptimalResult(trainX, trainY, predictX, predictY, resultList, feature, loop_condition, trainSelect, KinKNN=1):
    accuracy = 0.0
    for oplist in resultList:
        m = 0
        accuracy_temp = 0.0
        while m < loop_condition:
            m += 1
            accuracy_temp += check(trainX, trainY, predictX, predictY, oplist, feature, trainSelect, KinKNN)
        accuracy_temp /= loop_condition
        if accuracy < accuracy_temp:
            accuracy = accuracy_temp
            optimal_subset = oplist
        elif accuracy == accuracy_temp and len(oplist) < len(optimal_subset):
            optimal_subset = oplist
    DR = 1.0 - (1.0 * optimal_subset.count('1') / len(feature))
    return accuracy, DR, optimal_subset
