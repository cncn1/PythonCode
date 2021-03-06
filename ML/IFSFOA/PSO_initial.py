#!/usr/bin/python
# -*- coding:utf-8 -*-
import math
from numpy import *
from FSFOATOOL import *

loop_condition = 2  # 最起码要大于lifetime=15的值 ，因为播种一次age才曾1
trainX, trainy = loadData(
    'D:/LabDatas/processed_data/sonar/train_1.txt')  # trainX,trainy are all list
predictX, predicty = loadData('D:/LabDatas/processed_data/sonar/predict_1.txt')
initialization_parameters = [15, 12, 30, 0.05, 50]
# trainX,trainy=loadData('C:/Users/Administrator/Desktop/install file/processed_data/wine/train_1.txt')#trainX,trainy are all list
# predictX,predicty=loadData('C:/Users/Administrator/Desktop/install file/processed_data/wine/predict_1.txt')
# initialization_parameters = [15, 3, 6, 0.05, 50]


# 变量定义
num_tree_ini = 60  # 初始化时森林中tree的个数
initial_forest = []
area_limit_forest = []
num_fea_original = mat(trainX).shape[1]
feature = []  # 特征集合索引,特征集合的角标

for i in range(num_fea_original):
    feature.append(i)


class Tree:
    def __init__(self, tree_list, tree_age):
        self.age = tree_age
        self.list = tree_list


fs_num_fea = math.ceil(0.1 * num_fea_original)  # 向前选择的特征个数
half_fea = math.ceil(0.5 * num_fea_original)
bs_num_fea = random.randint(half_fea, num_fea_original - 1)  # 向后选择的特征个数，这个是随机选择的
print('fs_num_fea,half_fea,bs_num_fea:', fs_num_fea, half_fea, bs_num_fea)
ini_forest_const = 2.0 / 3
fs_num_tree_ini = int(num_tree_ini * (ini_forest_const))  # 初始化森林时向前选择的树的个数
bs_num_tree_ini = int(num_tree_ini * (1 - ini_forest_const))  # 初始化森林时向后选择的树的个数
print('fs_num_tree_ini:', fs_num_tree_ini, 'bs_num_tree_ini:', bs_num_tree_ini)
# 将森林中的树以字符串的形式初始化为全0
ini_str = ''
for i in range(num_fea_original):
    ini_str += '0'
print('ini_str', ini_str)
for i in range(num_tree_ini):
    initial_forest.append(ini_str)
# 将森林初始化为list为全0的字符串，age为0
for each_item in initial_forest:
    instance = Tree(each_item, 0)
    area_limit_forest.append(instance)
print('初始化森林的长度：', len(area_limit_forest))
for i in range(len(area_limit_forest)):
    print('刚刚初始化为list全0，age为0的area_limit_forest：', area_limit_forest[i].list, area_limit_forest[i].age)


# 产生随机数
def random_form(num_random):
    random_num = []
    j = 0
    while j < num_random:
        y = random.randint(0, num_fea_original - 1)
        if y not in random_num:
            random_num.append(y)
            j += 1
        else:
            continue
    return random_num


# 初始化时，将0翻成1
def ini_reverse(attri_reverse, area_limit_forest_single_tree):
    # after_reverse=[]
    temp = Tree(area_limit_forest_single_tree.list, area_limit_forest_single_tree.age)
    for i in range(len(attri_reverse)):
        const_value = 1
        new_string = index_replace(attri_reverse[i], temp.list, const_value)
        temp.list = new_string
    after_reverse = temp
    return after_reverse


# 2/3是forward selection,1/3是backward selection
def ini_PG(area_limit_forest):
    area_limit_forest_iniPG = []
    for i in range(len(area_limit_forest)):
        if (i < fs_num_tree_ini):
            attri_reverse = random_form(fs_num_fea)  # 每次选择不同的反转属性
            print('第', i, '棵树选择要反转的属性是：', attri_reverse)
            area_limit_forest_iniPG.append(ini_reverse(attri_reverse, area_limit_forest[i]))
        else:
            attri_reverse = random_form(bs_num_fea)
            print('第', i, '棵树选择要反转的属性是：', attri_reverse)
            area_limit_forest_iniPG.append(ini_reverse(attri_reverse, area_limit_forest[i]))
    return area_limit_forest_iniPG


area_limit_forest_iniPG = ini_PG(area_limit_forest)

for i in range(len(area_limit_forest_iniPG)):
    print('使用Ini_PG方法后area_limit_forest_iniPG：', area_limit_forest_iniPG[i].list, area_limit_forest_iniPG[i].age)
