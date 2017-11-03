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


# 根据信息熵就最优特征,代码引用机器学习实战决策树一章
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
        last_compare_subset_accuracy[optimal_feature_index[index]] = 1
        last_compare_subset_DR[optimal_feature_index[index]] = 1
        generate_num -= 1
    last_compare_subset_accuracy[optimal_feature_index[index - 1]] = 1
    return last_compare_subset_accuracy, last_compare_subset_DR


def OptimalResult(trainX, trainY, predictX, predictY, resultList, feature, trainSelect, KinKNN=1):
    accuracy = 0.0
    for index in xrange(len(resultList)):
        fea_list_CB = numtofea(resultList[index], feature)
        data_sample = read_data_fea(fea_list_CB, trainX)
        data_predict = read_data_fea(fea_list_CB, predictX)
        accuracy_temp = trainSelect(data_sample, trainY, data_predict, predictY, KinKNN)
        # if algorithm == 'KNN':
        #     accuracy_temp = train_knn(data_sample, trainY, data_predict, predictY, k)
        # elif algorithm == 'SVM':
        #     accuracy_temp = train_svm(data_sample, trainY, data_predict, predictY)
        # elif algorithm == 'J48':
        #     accuracy_temp = train_tree(data_sample, trainY, data_predict, predictY)
        if accuracy < accuracy_temp:
            accuracy = accuracy_temp
            optimal_subset = resultList[index]
        elif accuracy == accuracy_temp and len(resultList[index]) < len(optimal_subset):
            optimal_subset = resultList[index]
    DR = 1 - (1.0 * optimal_subset.count(1) / len(feature))
    return accuracy, DR, optimal_subset


if __name__ == '__main__':
    optimal_area = [[1, 1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1]]
    s1, s2 = GroupSelection(optimal_area, 7, 4)
    print s1, s2
    # myDat, labels = createDataSet()
    # optimalFeature = chooseBestFeatureToSplit(myDat)
    # print optimalFeature
