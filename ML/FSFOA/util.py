#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import FSFOATOOL as tools


# initialization_parameters: life time, LSC局部播种特征数, GSC全局播种特征数, transfer rate， area
def loadData(fileName):
    path = 'D:/LabDatas/FSFOA_Data/'
    if fileName == 'ionosphere':
        trainX, trainY = tools.loadData(path + 'ionosphere/train_1.txt')
        predictX, predictY = tools.loadData(path + 'ionosphere/predict_1.txt')
        loop_condition = 10
        initialization_parameters = [15, 7, 15, 0.05, 50]
    elif fileName == 'cleveland':
        trainX, trainY = tools.loadData(path + 'cleveland/train_70.txt')
        predictX, predictY = tools.loadData(path + 'cleveland/predict_30.txt')
        loop_condition = 10
        initialization_parameters = [15, 3, 6, 0.05, 50]
    elif fileName == 'wine':
        trainX, trainY = tools.loadData(path + 'wine/train_1.txt')
        predictX, predictY = tools.loadData(path + 'wine/predict_1.txt')
        loop_condition = 10
        initialization_parameters = [15, 3, 6, 0.05, 50]
    elif fileName == 'sonar':
        trainX, trainY = tools.loadData(path + 'sonar/train_701.txt')
        predictX, predictY = tools.loadData(path + 'sonar/predict_301.txt')
        loop_condition = 10
        initialization_parameters = [15, 12, 30, 0.05, 50]
    elif fileName == 'srbct':
        trainX, trainY = tools.loadData(path + 'SRBCT/train_703.txt')
        predictX, predictY = tools.loadData(path + 'SRBCT/predict_303.txt')
        loop_condition = 4
        initialization_parameters = [15, 6, 7, 0.05, 50]
    elif fileName == 'segmentation':
        trainX, trainY = tools.loadData(path + 'segmentation/train_1.txt')
        predictX, predictY = tools.loadData(path + 'segmentation/predict_1.txt')
        loop_condition = 3
        initialization_parameters = [15, 4, 9, 0.05, 50]
    elif fileName == 'vehicle':
        trainX, trainY = tools.loadData(path + 'vehicle/train_1.txt')
        predictX, predictY = tools.loadData(path + 'vehicle/predict_1.txt')
        loop_condition = 10
        initialization_parameters = [15, 4, 9, 0.05, 50]
    elif fileName == 'dermatology':
        trainX, trainY = tools.loadData(path + 'dermatology/train_701.txt')
        predictX, predictY = tools.loadData(path + 'dermatology/predict_301.txt')
        loop_condition = 10
        initialization_parameters = [15, 7, 15, 0.05, 50]
    elif fileName == 'heart':
        trainX, trainY = tools.loadData(path + 'heart/train_1.txt')
        predictX, predictY = tools.loadData(path + 'heart/predict_1.txt')
        loop_condition = 10
        initialization_parameters = [15, 3, 6, 0.05, 50]
    elif fileName == 'glass':
        trainX, trainY = tools.loadData(path + 'glass/train_1.txt')
        predictX, predictY = tools.loadData(path + 'glass/predict_1.txt')
        loop_condition = 10
        initialization_parameters = [15, 2, 4, 0.05, 50]
    elif fileName == 'arcene':  # uci 特征选择 二分类 数据集
        trainX, trainY = tools.loadData(path + 'arcene/arcene_train.txt', '\t')
        predictX, predictY = tools.loadData(path + 'arcene/arcene_valid.txt', '\t')
        loop_condition = 10
        initialization_parameters = [15, 2, 4, 0.05, 50]
    return trainX, trainY, predictX, predictY, loop_condition, initialization_parameters


# 数组反转方法
def revers(s, indexList, flag=1):
    if flag == 1:
        for index in indexList:
            s[index] = (s[index] + 1) % 2
    else:
        s[indexList] = (s[indexList] + 1) % 2
    return s


# 从feature_length个特征中产生一组随机数(num_random个)
def random_form(num_random, feature_length):
    random_num = []
    j = 0
    while j < num_random:
        y = np.random.randint(0, feature_length)
        if y not in random_num:
            random_num.append(y)
            j += 1
        else:
            continue
    return random_num


if __name__ == '__main__':
    inputDict = {'arcene': 'arcene', 'sonar': 'sonar'}
    for i in inputDict:
        trainX, trainY, predictX, predictY = loadData(i)  # trainX,trainY are all list
        datas = [trainX, trainY, predictX, predictY]
        # with open("E:/%s.txt" % i, "w") as f:
        #     for data in datas:
        #         f.write("%s\n" % data)
        #     f.close
        # print trainX, '\n', trainY, '\n', predictX, '\n', predictY
