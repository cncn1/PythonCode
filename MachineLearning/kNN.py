# coding=utf-8
from numpy import *
import operator

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1],[4,4.2],[4,4.3],[5,4.8]])
    labels = ['A', 'A', 'B', 'B','CT','CT','CT']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    print distances
    sortedDistIndicies = distances.argsort()  # argsort是排序,将元素按照由小到大的顺序返回下标，比如([3,1,2]),它返回的就是([1,2,0])
    print sortedDistIndicies
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1   # get是取字典里的元素，如果之前这个voteIlabel是有的，那么就返回字典里这个voteIlabel里的值，
                                                                     # 如果没有就返回0（后面写的），这行代码的意思就是算离目标点距离最近的k个点的类别，这个点是哪个类别哪个类别就加1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1),reverse=True)   #key=operator.itemgetter(1)的意思是按照字典里的第一个排序，{A:1,B:2},要按照第1个（'A','B'是第0个），即‘1’‘2’排序。reverse=True是降序排序
    print sortedClassCount
    return sortedClassCount[0][0]


group, labels = createDataSet()
print classify0([4.2, 4.9], group, labels, 3)
