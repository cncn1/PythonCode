#!/usr/bin/python
# -*- coding:utf-8 -*-
from copy import deepcopy
from FSFOATOOL import *


# 局部播种
def reverse_binary_LSC(vice_verse_attri, area_limit_forest_iniPG):
    after_reverse = []
    area_limit_forest_age0 = []
    for area_limit_forest_tree in area_limit_forest_iniPG:
        if area_limit_forest_tree.age == 0:
            # 将森林中age为0的树放入area_limit_forest_age0准备局部播种。
            area_limit_forest_age0.append(deepcopy(area_limit_forest_tree))
        else:
            continue
    for area_limit_forest_tree in area_limit_forest_iniPG:
        area_limit_forest_tree.age += 1
    for tree_age0 in area_limit_forest_age0:
        for each_attri in vice_verse_attri:
            temp_tree_age0 = deepcopy(tree_age0)
            temp_tree_age0.list = revers(temp_tree_age0.list, each_attri, 0)
            after_reverse.append(deepcopy(temp_tree_age0))
    return after_reverse


# 更新策略
def select_trees(trainX, trainY, predictX, predictY, life_time, area, feature, area_limit_forest_iniPG, trainSelect,
                 KinKNN=1):
    selected_trees = []  # 候选森林中的树
    acc = []
    acc_omit_index = []  # 存的是acc中前num_extra的最小值的角标
    age_exceed_lifetime_index = []  # age值超过lifetime的索引号
    # 将森林中年龄值大于年龄上限的树从森林中移除，放入候选森林
    for i in xrange(len(area_limit_forest_iniPG)):
        if area_limit_forest_iniPG[i].age > life_time:
            selected_trees.append(area_limit_forest_iniPG[i])
            age_exceed_lifetime_index.append(i)
    delete_together(age_exceed_lifetime_index, area_limit_forest_iniPG)
    # 如果原森林中剩余树的数量仍超出区域上限值，则根据树的适应度值（分类准确率）从小到大依次移除多余的树，并将这些移除的树放到候选森林中
    if len(area_limit_forest_iniPG) > area:
        # 遍历area_limit_forest_iniPG中剩下的树带入分类器（eg knn）算分类准确率，准确率低的放入候选区直至area_limit_forest_iniPG的长度为are_limit的值为止
        num_extra = len(area_limit_forest_iniPG) - area
        for limit_tree in area_limit_forest_iniPG:
            fea_list = numtofea(limit_tree.list, feature)
            if len(fea_list) > 0:
                data_sample = read_data_fea(fea_list, trainX)
                data_predict = read_data_fea(fea_list, predictX)
                acc.append(trainSelect(data_sample, trainY, data_predict, predictY, KinKNN))  # 每棵树的准确率存在acc中
            else:
                acc.append(0)
        for i in xrange(num_extra):
            acc_min = min(acc)
            acc_min_index = acc.index(acc_min)
            acc[acc_min_index] = 100
            acc_omit_index.append(acc_min_index)
        for each_item in acc_omit_index:
            selected_trees.append(deepcopy(area_limit_forest_iniPG[each_item]))
        delete_together(acc_omit_index, area_limit_forest_iniPG)
    return selected_trees


# FSFOA Global_seeding（全局播种）
def reverse_binary_GSC_FSFOA(transferRate, candidate_area, feature_length, GSC):
    after_reverse = []
    selected_tree_area = []  # 从候选区中挑出来进行反转的树
    num_percent_transfer = int(len(candidate_area) * transferRate)
    j = 0
    while j < num_percent_transfer:
        y = random.randint(0, len(candidate_area))
        if candidate_area[y] not in selected_tree_area:
            selected_tree_area.append(candidate_area[y])
            j = j + 1
        else:
            continue
    for selected_tree in selected_tree_area:
        vice_verse_attri_GSC = random_form(GSC, feature_length)  # 全局播种特征的集合
        selected_tree.list = revers(selected_tree.list, vice_verse_attri_GSC)
        after_reverse.append(deepcopy(selected_tree))
    return after_reverse


# EFSFOA Global_seeding（全局播种）
def reverse_binary_GSC_EFSFOA(transferRate, candidate_area, feature_length, GSC0):
    after_reverse = []
    selected_tree_area = []  # 从候选区中挑出来进行反转的树
    num_percent_transfer = int(len(candidate_area) * transferRate)
    j = 0
    while j < num_percent_transfer:
        y = random.randint(0, len(candidate_area))
        if candidate_area[y] not in selected_tree_area:
            selected_tree_area.append(candidate_area[y])
            j = j + 1
        else:
            continue
    for selected_tree in selected_tree_area:
        # GSC受退火函数启发 温度退火函数 可以让GSC有更广泛的搜索空间并利用了原始数据结构age动态变化的特点
        # 假设时刻t的温度用T(t)来表示, 则经典模拟退火算法的降温方式为:T(t) = T0 / lg(t + 1) 而 快速模拟退火算法的降温方式为：T(t) = T0 / (t + 1)  t从0开始取值
        GSC = int(GSC0 / (selected_tree.age + 1))
        vice_verse_attri_GSC = random_form(GSC, feature_length)  # 全局播种特征的集合
        selected_tree.list = revers(selected_tree.list, vice_verse_attri_GSC)
        after_reverse.append(deepcopy(selected_tree))
    return after_reverse
