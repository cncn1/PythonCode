#!/usr/bin/python
# -*- coding:utf-8 -*-
from numpy import *


# 简单的数据预处理（如用平均值代替缺失值）
def data_Preprocess(filename):
    numfeat = len(open(filename).readline().split('\t')) - 1
    dataMat = [];
    labelMat = []
    fr = open(filename)
    rownum = 0
    for line in fr.readlines():
        rownum += 1
        ArrLine = []
        curLine = line.strip().split('\t')
        for i in range(numfeat):
            print(curLine[i])
            print(curLine[i].isdigit())
            print(type(curLine[i]))
            if not curLine[i].isdigit():
                # print('this is not a number')
                print('the row number is :', rownum)
                print('the col number is : ', i)
                print('####################')
                mean_i = round(mean(mat(dataMat)[0:rownum - 1, i]))
                ArrLine.append(mean_i)
            else:
                ArrLine.append(float(curLine[i]))
        dataMat.append(ArrLine)
        # print(ArrLine)
        if curLine[-1].isdigit():
            labelMat.append(curLine[-1])
        else:
            print('this line miss the label:', rownum)
            print('--------------------')
    return dataMat, labelMat


datamat, labelmat = data_Preprocess('./SRBCT_processed.data')
fout = open('./srbct_processed.txt', 'w')
col_num, row_num = mat(datamat).shape
for i in range(col_num):
    for j in range(row_num):
        fout.write(str(mat(datamat)[i, j]))
        fout.write(',')
    fout.write(str(labelmat[i]))
    fout.write('\n')
fout.close()

print(mat(datamat))
print('_______________')
print(labelmat)
print(mat(datamat).shape)
