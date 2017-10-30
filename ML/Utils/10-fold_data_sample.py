#!/usr/bin/python
# -*- coding:utf-8 -*-
from numpy import *
from collections import defaultdict


def load_date_10_fold(filename):
    label = []
    numFeat = len(open(filename).readline().split(',')) - 1  # 样本中特征的个数
    fr = open(filename)
    classData = defaultdict(list)
    for line in fr.readlines():
        xi = []
        curLine = line.strip().split(',')
        for i in range(numFeat):
            xi.append(float(curLine[i]))  # 将第i个特征添加到xi
        classData[curLine[-1]].append(xi)
        label.append((curLine[-1]))
    return classData, label


def data_sample(labelMat, class_data):
    data_sam = defaultdict(list)
    cla_data = []
    for i in set(labelMat):
        for j in range(len(class_data[i])):
            b = [float(i)]
            cla_data.append((class_data[i])[j] + b)
    for k in range(len(cla_data)):
        data_sam[str(k % 10)].append(cla_data[k])
    return data_sam


classdata, label = load_date_10_fold('./train_70.txt')
data_sam = data_sample(label, classdata)
file_name = ['predict_1.txt', 'predict_2.txt', 'predict_3.txt', 'predict_4.txt',
             'predict_5.txt', 'predict_6.txt', 'predict_7.txt', 'predict_8.txt', 'predict_9.txt', 'predict_10.txt']
label_num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
n = 0
for k in label_num:
    fout = open(file_name[int(k)], 'w')
    for i in range(len(data_sam[k])):
        for j in range(len((data_sam[k])[i]) - 1):
            fout.write(str(((data_sam[k])[i])[j]))
            fout.write(',')
            n = j
        fout.write(str(((data_sam[k])[i])[n + 1]))
        fout.write('\n')
    fout.close()
train_label_1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
fout_1 = open('train_1.txt', 'w')
for k in train_label_1:
    for i in range(len(data_sam[k])):
        for j in range(len((data_sam[k])[i]) - 1):
            fout_1.write(str(data_sam[k][i][j]))
            fout_1.write(',')
            n = j
        fout_1.write(str(data_sam[k][i][n + 1]))
        fout_1.write('\n')
fout_1.close()
train_label_2 = ['0', '2', '3', '4', '5', '6', '7', '8', '9']
fout = open('train_2.txt', 'w')
for k in train_label_2:
    for i in range(len(data_sam[k])):
        for j in range(len(data_sam[k][i]) - 1):
            fout.write(str(data_sam[k][i][j]))
            fout.write(',')
            n = j
        fout.write(str(data_sam[k][i][n + 1]))
        fout.write('\n')
fout.close()
train_label_3 = ['0', '1', '3', '4', '5', '6', '7', '8', '9']
fout = open('train_3.txt', 'w')
for k in train_label_3:
    for i in range(len(data_sam[k])):
        for j in range(len(data_sam[k][i]) - 1):
            fout.write(str(data_sam[k][i][j]))
            fout.write(',')
            n = j
        fout.write(str(data_sam[k][i][n + 1]))
        fout.write('\n')
fout.close()
train_label_4 = ['0', '1', '2', '4', '5', '6', '7', '8', '9']
fout = open('train_4.txt', 'w')
for k in train_label_4:
    for i in range(len(data_sam[k])):
        for j in range(len(data_sam[k][i]) - 1):
            fout.write(str(data_sam[k][i][j]))
            fout.write(',')
            n = j
        fout.write(str(data_sam[k][i][n + 1]))
        fout.write('\n')
fout.close()
train_label_5 = ['0', '1', '2', '3', '5', '6', '7', '8', '9']
fout = open('train_5.txt', 'w')
for k in train_label_5:
    for i in range(len(data_sam[k])):
        for j in range(len(data_sam[k][i]) - 1):
            fout.write(str(data_sam[k][i][j]))
            fout.write(',')
            n = j
        fout.write(str(data_sam[k][i][n + 1]))
        fout.write('\n')
fout.close()
train_label_6 = ['0', '1', '2', '3', '4', '6', '7', '8', '9']
fout = open('train_6.txt', 'w')
for k in train_label_6:
    for i in range(len(data_sam[k])):
        for j in range(len(data_sam[k][i]) - 1):
            fout.write(str(data_sam[k][i][j]))
            fout.write(',')
            n = j
        fout.write(str(data_sam[k][i][n + 1]))
        fout.write('\n')
fout.close()
train_label_7 = ['0', '1', '2', '3', '4', '5', '7', '8', '9']
fout = open('train_7.txt', 'w')
for k in train_label_7:
    for i in range(len(data_sam[k])):
        for j in range(len(data_sam[k][i]) - 1):
            fout.write(str(data_sam[k][i][j]))
            fout.write(',')
            n = j
        fout.write(str(data_sam[k][i][n + 1]))
        fout.write('\n')
fout.close()
train_label_8 = ['0', '1', '2', '3', '4', '5', '6', '8', '9']
fout = open('train_8.txt', 'w')
for k in train_label_8:
    for i in range(len(data_sam[k])):
        for j in range(len(data_sam[k][i]) - 1):
            fout.write(str(data_sam[k][i][j]))
            fout.write(',')
            n = j
        fout.write(str(data_sam[k][i][n + 1]))
        fout.write('\n')
fout.close()
train_label_9 = ['0', '1', '2', '3', '4', '5', '6', '7', '9']
fout = open('train_9.txt', 'w')
for k in train_label_9:
    for i in range(len(data_sam[k])):
        for j in range(len(data_sam[k][i]) - 1):
            fout.write(str(data_sam[k][i][j]))
            fout.write(',')
            n = j
        fout.write(str(data_sam[k][i][n + 1]))
        fout.write('\n')
fout.close()
train_label_10 = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
fout = open('train_10.txt', 'w')
for k in train_label_10:
    for i in range(len(data_sam[k])):
        for j in range(len(data_sam[k][i]) - 1):
            fout.write(str(data_sam[k][i][j]))
            fout.write(',')
            n = j
        fout.write(str(data_sam[k][i][n + 1]))
        fout.write('\n')
fout.close()
