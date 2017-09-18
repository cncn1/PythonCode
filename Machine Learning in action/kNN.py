#!/usr/bin/python
# -*- coding:utf-8 -*-
from numpy import *
import operator


def createDataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()  # 返回从小到大排序的索引值
    classCount = {}
    for i in xrange(k):
        voteIlable = labels[sortedDistIndicies[i]]  #
        classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    group, labels = createDataset()
    print classify0([0, 0], group, labels, 3)
