#!/usr/bin/python
# -*- coding:utf-8 -*-
import util
import numpy as np
import math
import FSFOATOOL as tools

# trainX,trainY=loadData('C:/Users/Administrator/Desktop/install file/processed_data/wine/train_1.txt')#trainX,trainY are all list
# predictX,predictY=loadData('C:/Users/Administrator/Desktop/install file/processed_data/wine/predict_1.txt')
# initialization_parameters = [15, 3, 6, 0.05, 50]
# 变量定义
inputDict = {'arcene': 'arcene', 'sonar': 'sonar'}
trainX, trainY, predictX, predictY = util.loadData(inputDict['sonar'])  # trainX,trainY,predictX,predictY are all list
initialization_parameters = [15, 12, 30, 0.05, 50]
loop_condition = 2  # 最起码要大于lifetime=15的值 ，因为播种一次age才增1
num_tree_ini = 60  # 初始化时森林中tree的个数 ， 这里可以改进
initial_forest = []
area_limit_forest = []
num_fea_original = np.mat(trainX).shape[1]  # 特征长度
feature = []  # 特征集合索引,特征集合的角标
for i in range(num_fea_original):
    feature.append(i)


class Tree:
    def __init__(self, tree_list, tree_age):
        self.list = tree_list
        self.age = tree_age


# 初始化策略(这里上启发式),可以上决策树熵理论，不随机播特征，播数据（或是取子集kmeans++之后播）


'''
fs_num_fea = math.ceil(0.1 * num_fea_original)  # 向前选择的特征个数
half_fea = math.ceil(0.5 * num_fea_original)
bs_num_fea = random.randint(half_fea, num_fea_original - 1)  # 向后选择的特征个数，这个是随机选择的
print('fs_num_fea,half_fea,bs_num_fea:', fs_num_fea, half_fea, bs_num_fea)
ini_forest_const = 2.0 / 3.0
fs_num_tree_ini = int(num_tree_ini * (ini_forest_const))  # 初始化森林时向前选择的树的个数
bs_num_tree_ini = int(num_tree_ini * (1 - ini_forest_const))  # 初始化森林时向后选择的树的个数
print('fs_num_tree_ini:', fs_num_tree_ini, 'bs_num_tree_ini:', bs_num_tree_ini)
'''

# 将每棵树记录的特征以全0数组的形式初始化
initial_forest = [0] * num_fea_original
# 初始化森林
for each_item in xrange(num_tree_ini):
    instance = Tree(initial_forest, 0)
    area_limit_forest.append(instance)


# print '初始化森林的长度：', len(area_limit_forest)
# for i in range(len(area_limit_forest)):
#     print '刚刚初始化为list全0，age为0的area_limit_forest：', area_limit_forest[i].list, area_limit_forest[i].age


# 产生一组随机数
def random_form(num_random):
    random_num = []
    j = 0
    while j < num_random:
        y = np.random.randint(0, num_fea_original)
        if y not in random_num:
            random_num.append(y)
            j += 1
        else:
            continue
    return random_num


# 初始化,针对森林中的树构造对应的个体
def ini_reverse(attri_reverse, area_limit_forest_single_tree):
    tree = Tree(area_limit_forest_single_tree.list, area_limit_forest_single_tree.age)
    # for i in attri_reverse:
    # const_value = 1
    # util.revers(temp.list, i)
    # new_string = tools.index_replace(attri_reverse[i], temp.list, const_value)
    tree.list = util.revers(tree.list, attri_reverse)
    return tree


'''# 2/3是forward selection,1/3是backward selection'''
def ini_PG(area_limit_forest):
    area_limit_forest_iniPG = []
    for i in range(len(area_limit_forest)):
        '''原始算法,这里不要
        if (i < fs_num_tree_ini):
            attri_reverse = random_form(fs_num_fea)  # 每次选择不同的反转属性
            print('第', i, '棵树选择要反转的属性是：', attri_reverse)
            area_limit_forest_iniPG.append(ini_reverse(attri_reverse, area_limit_forest[i]))
        else:
            attri_reverse = random_form(bs_num_fea)
            print('第', i, '棵树选择要反转的属性是：', attri_reverse)
            area_limit_forest_iniPG.append(ini_reverse(attri_reverse, area_limit_forest[i]))'''
        init_feature_num = np.random.randint(0, num_fea_original)
        attri_reverse = random_form(init_feature_num)
        tree = ini_reverse(attri_reverse, area_limit_forest[i])
        # print tree.list
        # area_limit_forest_iniPG.append(ini_reverse(attri_reverse, area_limit_forest[i]))
        area_limit_forest_iniPG.append(tree)
        print area_limit_forest_iniPG[i].list
    return area_limit_forest_iniPG


area_limit_forest_iniPG = ini_PG(area_limit_forest)
for i in range(len(area_limit_forest_iniPG)):
    print '初始化树', area_limit_forest_iniPG[i].list, area_limit_forest_iniPG[i].age
