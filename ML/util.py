#!/usr/bin/python
# -*- coding:utf-8 -*-
from numpy import *


def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 每一列的最小值
    maxVals = dataSet.max(0)  # 每一列的最大值
    ranges = maxVals - minVals  # 幅度
    normDataSet = zeros(shape(dataSet))  # 创建一个一样规模的零数组
    m = dataSet.shape[0]  # 取数组的行
    normDataSet = dataSet - tile(minVals, (m, 1))  # 减去最小值
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide
    # 再除以幅度值，实现归一化，tile功能是创建一定规模的指定数组
    return normDataSet, ranges, minVals

