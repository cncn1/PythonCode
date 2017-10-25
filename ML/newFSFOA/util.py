#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import FSFOATOOL as tools


def loadData(fileName):
    path = 'D:/LabDatas/FSFOA_Data/'
    if fileName == 'sonar':
        trainX, trainY = tools.loadData(path + 'sonar/train_1.txt')
        predictX, predictY = tools.loadData(path + 'sonar/predict_1.txt')
    if fileName == 'arcene':  # uci 特征选择 二分类 数据集
        trainX, trainY = tools.loadData(path + 'arcene_train.txt', '\t')
        predictX, predictY = tools.loadData(path + 'arcene_valid.txt', '\t')
    return trainX, trainY, predictX, predictY


# 数组反转方法
def revers(index):
    s = [1] * 9
    index = index
    s[index] = (s[index] + 1) % 2
    print s


if __name__ == '__main__':
    inputDict = {'arcene': 'arcene', 'sonar': 'sonar'}
    for i in inputDict:
        trainX, trainY, predictX, predictY = loadData(i)  # trainX,trainY are all list
        datas = [trainX, trainY, predictX, predictY]
        with open("E:/%s.txt" % i, "w") as f:
            for data in datas:
                f.write("%s\n" % data)
            f.close
            # print trainX, '\n', trainY, '\n', predictX, '\n', predictY
