#!/usr/bin/python
# -*- coding:utf-8 -*-
from numpy import mat
from numpy import random
from sklearn import neighbors
from sklearn import svm
from sklearn import tree


# 根据选出的特征返回对应的特征编号
def numtofea(num, fea_list):
    feature = []
    for i in xrange(len(num)):
        if num[i] == 1:
            feature.append(fea_list[i])
        else:
            continue
    return feature


# 数组反转方法,类似模2除,针对对应特征0和1的反转方法,flag=1时反转所给定列表中的全部特征,flag不等于1时反转给定位置的特征
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
        y = random.randint(0, feature_length)
        if y not in random_num:
            random_num.append(y)
            j += 1
        else:
            continue
    return random_num


# 根据所选特征选择子数据集
def read_data_fea(fea_list, dataset):
    dataMat = mat(dataset)
    col = dataMat.shape[0]  # 行号
    data_sample = []
    for i in xrange(col):
        col_i = []
        for j in fea_list:
            col_i.append(dataMat[i, j])
        data_sample.append(col_i)
    return data_sample


# 在指定角标索引的位置将字符串进行替换
def index_replace(index, replace_string, const_value):
    new_string = ''
    for i in xrange(len(replace_string)):
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
    return 1.0 - 1.0 * num / len(label_train)


def train_knn(data_train, label_train, data_pre, label_pre, k=1):
    clf = neighbors.KNeighborsClassifier(n_neighbors=k)  # 创建分类器对象
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
