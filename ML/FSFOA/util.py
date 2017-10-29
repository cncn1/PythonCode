#!/usr/bin/python
# -*- coding:utf-8 -*-
import datetime
import ADAFSFOA_CORE

# 读取文件，返回list结构
def loadDataBase(filename, delimiter=','):
    numFeat = len(open(filename).readline().split(delimiter)) - 1
    feature = []
    label = []
    fr = open(filename)
    for line in fr.readlines():
        xi = []
        curline = line.strip().split(delimiter)
        for i in range(numFeat):
            # print curline[i]
            xi.append(float(curline[i]))
        feature.append(xi)
        label.append(float(curline[-1]))
    return feature, label


# 根据文件号,拼出37折对应的后缀名称01,02，...，10这种形式
def transStr(eachfileNum0):
    return str(int(eachfileNum0 / 10)) + str((eachfileNum0 % 10))


# 根据实验具体内容和对应的文件号,拼出对应的具体文件名
def transFileName(labName, eachfileNum):
    if labName == 1:  # 代表正常的实验
        trainFileName = 'train_' + str(eachfileNum) + '.txt'
        predictFileName = 'predict_' + str(eachfileNum) + '.txt'
    elif labName == 2:  # 代表2折实验
        trainFileName = 'train_' + str(eachfileNum) + '_2fold.txt'
        predictFileName = 'predict_' + str(eachfileNum) + '_2fold.txt'
    elif labName == 37:  # 代表37折实验
        trainFileName = 'train_7' + transStr(eachfileNum) + '.txt'
        predictFileName = 'predict_3' + transStr(eachfileNum) + '.txt'
    return trainFileName, predictFileName


# initialization_parameters: life time, LSC局部播种特征数, GSC全局播种特征数, transfer rate， area
def loadData(groupName, labName, eachfileNum):
    path = 'D:/LabDatas/FSFOA_Data/' + groupName  # 文件夹路径名
    eachTrainFile, eachPredictFile = transFileName(labName, eachfileNum)
    trainFile = path + '/' + eachTrainFile
    predictFile = path + '/' + eachPredictFile
    print '读取文件' + trainFile + '\t' + '读取文件' + predictFile
    if groupName == 'ionosphere':
        trainX, trainY = loadDataBase(trainFile)
        predictX, predictY = loadDataBase(predictFile)
        loop_condition = 2
        initialization_parameters = [15, 7, 15, 0.05, 50]
    elif groupName == 'cleveland':
        trainX, trainY = loadDataBase(trainFile)
        predictX, predictY = loadDataBase(predictFile)
        loop_condition = 2
        initialization_parameters = [15, 3, 6, 0.05, 50]
    elif groupName == 'wine':
        trainX, trainY = loadDataBase(trainFile)
        predictX, predictY = loadDataBase(predictFile)
        loop_condition = 2
        initialization_parameters = [15, 3, 6, 0.05, 50]
    elif groupName == 'sonar':
        trainX, trainY = loadDataBase(trainFile)
        predictX, predictY = loadDataBase(predictFile)
        loop_condition = 2
        initialization_parameters = [15, 12, 30, 0.05, 50]
    elif groupName == 'srbct':
        trainX, trainY = loadDataBase(trainFile)
        predictX, predictY = loadDataBase(predictFile)
        loop_condition = 4
        initialization_parameters = [15, 6, 7, 0.05, 50]
    elif groupName == 'segmentation':
        trainX, trainY = loadDataBase(trainFile)
        predictX, predictY = loadDataBase(predictFile)
        loop_condition = 3
        initialization_parameters = [15, 4, 9, 0.05, 50]
    elif groupName == 'vehicle':
        trainX, trainY = loadDataBase(trainFile)
        predictX, predictY = loadDataBase(predictFile)
        loop_condition = 2
        initialization_parameters = [15, 4, 9, 0.05, 50]
    elif groupName == 'dermatology':
        trainX, trainY = loadDataBase(trainFile)
        predictX, predictY = loadDataBase(predictFile)
        loop_condition = 2
        initialization_parameters = [15, 7, 15, 0.05, 50]
    elif groupName == 'heart':
        trainX, trainY = loadDataBase(trainFile)
        predictX, predictY = loadDataBase(predictFile)
        loop_condition = 2
        initialization_parameters = [15, 3, 6, 0.05, 50]
    elif groupName == 'glass':
        trainX, trainY = loadDataBase(trainFile)
        predictX, predictY = loadDataBase(predictFile)
        loop_condition = 2
        initialization_parameters = [15, 2, 4, 0.05, 50]
    elif groupName == 'arcene':  # uci 特征选择 二分类 数据集
        trainX, trainY = loadDataBase(trainFile, '\t')
        predictX, predictY = loadDataBase(predictFile, '\t')
        loop_condition = 2
        initialization_parameters = [15, 2, 4, 0.05, 50]
    return trainX, trainY, predictX, predictY, loop_condition, initialization_parameters


# 向文件中输出实验结果
def print_to_file(algorithmName, dataSetName, labName, accuracy_mean, DR_mean):
    out_file_name = dataSetName + '_' + str(labName)
    out_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    out_result = out_time + '\n' + str(float('%.2f' % accuracy_mean)) + '\t' + str(float('%.2f' % DR_mean))
    with open("E:/AlgorithmOut/" + algorithmName + "/%s.txt" % out_file_name, "a") as f:
        f.write("%s\n\n" % out_result)



if __name__ == '__main__':
    # 变量定义
    inputDict = {'ionosphere': ['ionosphere', [1, 1, 10, 2, 1, 2, 37, 1, 10]], 'cleveland': ['cleveland', [37, 1, 1]],
                 'wine': ['wine', [1, 1, 10, 2, 1, 2, 37, 1, 9]], 'sonar': ['sonar', [1, 1, 10, 2, 1, 2, 37, 1, 10]],
                 'srbct': ['srbct', [37, 1, 10]], 'segmentation': ['segmentation', [1, 1, 10]],
                 'vehicle': ['vehicle', [1, 1, 10, 2, 1, 2, 37, 1, 1]],
                 'dermatology': ['dermatology', [1, 1, 10, 37, 1, 10]], 'heart': ['heart', [1, 1, 10, 2, 1, 2]],
                 'glass': ['glass',[1, 1, 10, 2, 1, 2, 37, 1, 1]], 'arcene': ['arcene', [1, 1, 1]]}
    # trainX,trainY,predictX,predictY are all list
    for key in inputDict:
        dataSet = inputDict[key]
        loop0 = len(dataSet[1]) / 3  # 实验组数
        for loop in xrange(loop0):
            labName = dataSet[1][(loop * 3)]  # 每组实验具体内容
            labTimes = dataSet[1][(loop * 3) + 1]  # 每组实验重复次数
            fileNum = dataSet[1][(loop * 3 + 2)]  # 每组实验文件个数
            for times in xrange(labTimes):
                for eachfile in xrange(fileNum):
                    trainX, trainY, predictX, predictY, loop_condition, initialization_parameters = loadData(dataSet[0], labName, eachfile + 1)
                    optimalFeature = ADAFSFOA_CORE.chooseBestFeatureToSplit(trainX)
                    print optimalFeature