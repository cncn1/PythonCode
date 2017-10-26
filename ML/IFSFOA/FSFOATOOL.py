#!/usr/bin/python
# -*- coding:utf-8 -*-
from sklearn import neighbors
from sklearn import svm
from numpy import *  # mat
from sklearn import tree


# 读取文件，返回list结构
def loadData(filename):
    numFeat = len(open(filename).readline().split(',')) - 1
    feature = [];
    label = []
    fr = open(filename)
    for line in fr.readlines():
        xi = []
        curline = line.strip().split(',')
        for i in range(numFeat):
            xi.append(float(curline[i]))
        feature.append(xi)
        label.append(float(curline[-1]))
    return feature, label


def read_data_fea(fea_list, dataset):
    dataMat = mat(dataset)
    col = dataMat.shape[0]  # 行号
    data_sample = []
    for i in range(col):
        col_i = []
        for j in fea_list:
            col_i.append(dataMat[i, j])
        data_sample.append(col_i)
    return data_sample


# 在指定角标索引的位置将字符串进行替换
def index_replace(index, replace_string, const_value):
    new_string = ''
    for i in range(len(replace_string)):
        if i != index:
            new_string += replace_string[i]
        else:
            new_string += str(const_value)
    return new_string


# 给定索引列表delete_index，统一删除列表a中相应索引的元素
def delete_together(delete_index, a):
    for i in delete_index:
        a[i] = 'k'
    for i in range(len(delete_index)):
        a.remove('k')


# 预测标签和ground_true标签对比 算准确率
def acc_pre(label_pre, label_train):
    num = 0
    for i in range(len(label_pre)):
        if label_pre[i] != label_train[i]:
            num += 1
    return (1 - num / len(label_train))


def train_knn(data_train, label_train, data_pre, label_pre):
    clf = neighbors.KNeighborsClassifier(n_neighbors=1)  # 创建分类器对象
    clf.fit(data_train, label_train)  # 用训练数据拟合分类器模型搜索
    predict = clf.predict(data_pre)
    acc = acc_pre(predict, label_pre)  # 预测标签和ground_true标签对比 算准确率
    return acc


def train_svm(data_train, label_train, data_predict, label_predict):
    clf = svm.SVC()
    clf.fit(data_train, label_train)
    predict = clf.predict(data_predict)
    acc = acc_pre(predict, label_predict)
    return acc


def train_tree(data_train, label_train, data_pre, label_pre):
    # dot_data=StringIO()
    clf = tree.DecisionTreeClassifier()
    clf.fit(data_train, label_train)
    # tree.export_graphviz(clf,out_file=dot_data)
    # graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf('wine.pdf')
    predict = clf.predict(data_pre)
    acc = acc_pre(predict, label_pre)
    return acc


def numtofea(num, fea_list):
    feature = []
    for i in range(len(num)):
        if num[i] == '1':
            feature.append(fea_list[i])
        else:
            continue
    return feature
