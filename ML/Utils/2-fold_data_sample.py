#!/usr/bin/python
# -*- coding:utf-8 -*-
from numpy import *
from collections import defaultdict


def load_date_2_fold(filename):
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
        data_sam[str(k % 2)].append(cla_data[k])
    return data_sam


classdata, label = load_date_2_fold('./glass_processed.txt')
data_sam = data_sample(label, classdata)
file_name = ['predict_1_2fold.txt', 'predict_2_2fold.txt']
label_num = ['0', '1']
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

train_label_1 = ['1']
fout_1 = open('train_1_2fold.txt', 'w')
for k in train_label_1:
    for i in range(len(data_sam[k])):
        for j in range(len((data_sam[k])[i]) - 1):
            fout_1.write(str(data_sam[k][i][j]))
            fout_1.write(',')
            n = j
        fout_1.write(str(data_sam[k][i][n + 1]))
        fout_1.write('\n')
fout_1.close()

train_label_1 = ['0']
fout_1 = open('train_2_2fold.txt', 'w')
for k in train_label_1:
    for i in range(len(data_sam[k])):
        for j in range(len((data_sam[k])[i]) - 1):
            fout_1.write(str(data_sam[k][i][j]))
            fout_1.write(',')
            n = j
        fout_1.write(str(data_sam[k][i][n + 1]))
        fout_1.write('\n')
fout_1.close()
