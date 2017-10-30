#!/usr/bin/python
# -*- coding:utf-8 -*-
from numpy import *
from collections import defaultdict


def load_date_37_fold(filename):
    label = []
    numFeat = len(open(filename).readline().split(',')) - 1
    fr = open(filename)
    classData = defaultdict(list)
    for line in fr.readlines():
        xi = []
        curLine = line.strip().split(',')
        for i in range(numFeat):
            xi.append(float(curLine[i]))
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


classData, label = load_date_37_fold('./wine_processed.txt')
data_sam = data_sample(label, classData)
data_num_70 = ['0', '1', '2', '3', '4', '5', '6']
data_num_30 = ['7', '8', '9']
data_sam_70 = []
data_sam_30 = []
for i in data_num_70:
    for j in range(len(data_sam[i])):
        data_sam_70.append((data_sam[i])[j])

for i in data_num_30:
    for j in range(len(data_sam[i])):
        data_sam_30.append((data_sam[i])[j])

print(mat(data_sam_70).shape)
print(mat(data_sam_30).shape)
col_1, row_1 = mat(data_sam_70).shape
col_2, row_2 = mat(data_sam_30).shape
fout = open('./train_70.txt', 'w')
n1 = 0
for i in range(col_1):
    for j in range(row_1 - 1):
        fout.write(str(mat(data_sam_70)[i, j]))
        fout.write(',')
        n1 = j
    fout.write(str(mat(data_sam_70)[i, n1 + 1]))
    fout.write('\n')
fout.close()

fout = open('./predict_30.txt', 'w')
n2 = 0
for i in range(col_2):
    for j in range(row_2 - 1):
        fout.write(str(mat(data_sam_30)[i, j]))
        fout.write(',')
        n2 = j
    fout.write(str(mat(data_sam_30)[i, n2 + 1]))
    fout.write('\n')
fout.close()
