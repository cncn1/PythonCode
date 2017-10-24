#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import FSFOATOOL as tools

def loadData(fileName):
    if fileName == 'arcene':  # uci 特征选择 二分类 数据集
        # x0path = '..//Datas/uci/arcene/arcene.data'
        # y0path = '..//Datas/uci/arcene/arcene.labels'
        # x0 = np.loadtxt(x0path, dtype=int, delimiter=' ')
        # y0 = np.loadtxt(y0path, dtype=int, delimiter=' ')
        # y0 = np.mat(y0).T
        trainX, trainy = tools.loadData()
        predictX, predicty = tools.loadData()
    return trainX, trainy, predictX, predicty
