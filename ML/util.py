#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np


def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 每一列的最小值
    maxVals = dataSet.max(0)  # 每一列的最大值
    ranges = maxVals - minVals  # 幅度
    normDataSet = np.zeros(np.shape(dataSet))  # 创建一个一样规模的零数组
    m = dataSet.shape[0]  # 取数组的行
    normDataSet = dataSet - np.tile(minVals, (m, 1))  # 减去最小值
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # element wise divide
    # 再除以幅度值，实现归一化，tile功能是创建一定规模的指定数组
    return normDataSet


def heart_type(s):
    it = {'1': 1, '2': -1}
    return it[s]


def loadData(fileName, path='', delimiter='\t'):
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
    else:
        data = np.loadtxt(path, dtype=float, delimiter=delimiter)
        x0, y0 = np.split(data, (-1,), axis=1)
    return x0, y0


def loadDataSet(fileName, symbol):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split(symbol)
        dataMat.append(map(float, lineArr[:-1]))
        labelMat.append(float(lineArr[-1]))
    return dataMat, labelMat
