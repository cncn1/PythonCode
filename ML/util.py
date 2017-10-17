#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import bz2file
from libsvm.python.svm import *
from libsvm.python.svmutil import *

def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 每一列的最小值
    maxVals = dataSet.max(0)  # 每一列的最大值
    ranges = maxVals - minVals  # 幅度
    normDataSet = np.zeros(np.shape(dataSet))  # 创建一个一样规模的零数组
    m = dataSet.shape[0]  # 取数组的行
    normDataSet = dataSet - np.tile(minVals, (m, 1))  # 减去最小值
    normDataSet = normDataSet / (np.tile(ranges, (m, 1)) + 1e-8)  # element wise divide
    # 再除以幅度值，实现归一化，tile功能是创建一定规模的指定数组
    return normDataSet


def heart_type(s):
    it = {'1': 1, '2': -1}
    return it[s]


def loadData(fileName, path='', delimiter='\t', low=-1, len=1000):
    labDatasPath = 'D:\\LabDatas\\'
    if fileName == 'parkinsons':
        path = '..//Datas/uci/parkinsons/parkinsons.data'  # 数据文件路径
        data = np.loadtxt(path, dtype=float, delimiter=',')
        x0, y0 = np.split(data, (-1,), axis=1)
    elif fileName == 'heart':
        path = '..//Datas/uci/heart/heart.dat'
        data = np.loadtxt(path, dtype=float, delimiter=' ', converters={13: heart_type})
        x0, y0 = np.split(data, (-1,), axis=1)
    elif fileName == 'testSet':
        path = '..//Datas/testSet.txt'
        data = np.loadtxt(path, dtype=float, delimiter='\t')
        x0, y0 = np.split(data, (-1,), axis=1)
    elif fileName == 'testSetRBF':
        path = '..//Datas/testSetRBF.txt'
        data = np.loadtxt(path, dtype=float, delimiter='\t')
        x0, y0 = np.split(data, (-1,), axis=1)
    elif fileName == 'testSetRBF2':
        path = '..//Datas/testSetRBF2.txt'
        data = np.loadtxt(path, dtype=float, delimiter='\t')
        x0, y0 = np.split(data, (-1,), axis=1)
    elif fileName == 'arcene':  # uci 特征选择 二分类 数据集
        x0path = '..//Datas/uci/arcene/arcene.data'
        y0path = '..//Datas/uci/arcene/arcene.labels'
        x0 = np.loadtxt(x0path, dtype=int, delimiter=' ')
        y0 = np.loadtxt(y0path, dtype=int, delimiter=' ')
        y0 = np.mat(y0).T
    elif fileName == 'alpha':
        # pathX = labDatasPath + "Large Scale Data FTP\\alpha\\alpha_train.dat.bz2"
        # x0 = loadBz2fileDataSet(pathX, ' ')
        pathX = labDatasPath + "Large Scale Data FTP\\alpha\\alpha_train.dat"
        x0 = loadDataSet(pathX, ' ', low, (low + len - 1))
        x0 = np.mat(np.array(x0))
        pathY = labDatasPath + "Large Scale Data FTP\\alpha\\alpha_train.lab.bz2"
        y0 = loadBz2fileDataSet(pathY, ' ', low, (low + len - 1))
        y0 = np.mat(np.array(y0))
        y0 = np.mat(y0)
        # print x0[0:5], '\n', y0[0:5]
        # print x0.shape, ':', y0.shape
    elif fileName == 'news20':
        path = labDatasPath + "libsvmData\\news20\\news20.binary"
        train_label, train_pixel = svm_read_problem(path)
        print train_label
        # print train_label
        # print x[0:5]
    else:
        data = np.loadtxt(path, dtype=float, delimiter=delimiter)
        x0, y0 = np.split(data, (-1,), axis=1)
    return x0, y0


def loadBz2fileDataSet(filePath, symbol, low=-1, high=1000):  # 当给定low>=0时，是一种省内存费时间的切片方法
    dataMat = []
    fr = bz2file.open(filePath, "r")
    if low < 0:
        for line in fr.readlines():
            lineArr = line.strip().split(symbol)
            dataMat.append(map(float, lineArr))
    else:
        count = 1
        for line in fr.readlines():
            if count >= low:
                lineArr = line.strip().split(symbol)
                dataMat.append(map(float, lineArr))
            if count == high:
                break
            count += 1
    return dataMat


def loadDataSet(filePath, symbol, low=-1, high=1000):  # 当给定low>=0时，是一种省内存费时间的切片方法
    dataMat = []
    fr = open(filePath)
    if low < 0:
        for line in fr.readlines():
            lineArr = line.strip().split(symbol)
            dataMat.append(map(float, lineArr))
    else:
        count = 1
        for line in fr.readlines():
            if count >= low:
                lineArr = line.strip().split(symbol)
                dataMat.append(map(float, lineArr))
            if count == high:
                break
            count += 1
    return dataMat


def loadDataSets(fileName, symbol):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split(symbol)
        dataMat.append(map(float, lineArr[:-1]))
        labelMat.append(float(lineArr[-1]))
    return dataMat, labelMat
