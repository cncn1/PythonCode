#!/usr/bin/python
# -*- coding:utf-8 -*-
from FSFOATOOL import *
from PSO_initial import *


# local_seeding
def reverse_binary(vice_verse_attri, area_limit_forest_iniPG):  # area_limit_forest_age0[i]
    after_reverse = []
    area_limit_forest_age0 = []
    for i in xrange(len(area_limit_forest_iniPG)):
        if area_limit_forest_iniPG[i].age == 0:
            area_limit_forest_age0.append(deepcopy(area_limit_forest_iniPG[i]))  # 确保原始树送入reverse_binary反转产生新树后，原始树的age值加1。
        else:
            continue
    # for i in range(len(area_limit_forest_age0)):
    #     print('age0的age是：',areax_limit_forest_age0[i].list,area_limit_forest_age0[i].age,len(area_limit_forest_age0))
    for i in xrange(len(area_limit_forest_iniPG)):
        area_limit_forest_iniPG[i].age += 1
    for i in xrange(len(area_limit_forest_age0)):
        print '+1后age0的age是：', area_limit_forest_age0[i].list, area_limit_forest_age0[i].age, len(area_limit_forest_age0)
    for i in xrange(len(area_limit_forest_age0)):
        print '刚开始时的area_limit_forest_age0[', i, '].list和.age的值', area_limit_forest_age0[i].list,area_limit_forest_age0[i].age
        # print('area_limit_forest_age0[',i,'].age',area_limit_forest[i].age)
        print '需要反转的属性', vice_verse_attri
        for vice_index in xrange(vice_verse_attri):
            temp = Tree(area_limit_forest_age0[i].list, 0)
            # 属性反转
            temp.list[vice_index] =

            if temp.list[vice_verse_attri[k]] == '0':
                const_value = 1
                new_string = index_replace(vice_verse_attri[k], temp.list, const_value)
                temp.list = new_string
            # temp.list[vice_verse_attri[i]].re='1'
            # print('haha')
            # print('temp.list是不是字符串类型',isinstance(temp.list,str))temp.list是字符串类型
            else:
                const_value = 0
                new_string = index_replace(vice_verse_attri[k], temp.list, const_value)
                temp.list = new_string
            # temp.list[vice_verse_attri[i]] == '0'
            # temp.list=int(temp.list,2)
            # print('temp的list值和age值',temp.list,temp.age)
            after_reverse.append(temp)
            # area_limit_forest_age0[i].list=int(str0,2)
    return after_reverse


# population_limiting(改进后的)
def select_trees(area_limit_forest_iniPG):
    selected_trees = []
    acc = []
    acc_omit_index = []  # 存的是acc中前num_extra的最小值的角标
    age_exceed_lifetime_index = []  # age值超过lifetime的索引号
    if len(area_limit_forest_iniPG) <= initialization_parameters[4]:
        for i in range(len(area_limit_forest_iniPG)):
            if area_limit_forest_iniPG[i].age > initialization_parameters[0]:
                selected_trees.append(area_limit_forest_iniPG[i])
                age_exceed_lifetime_index.append(i)
                # area_limit_forest.remove(area_limit_forest[i])
        delete_together(age_exceed_lifetime_index, area_limit_forest_iniPG)
    else:
        for i in range(len(area_limit_forest_iniPG)):
            if area_limit_forest_iniPG[i].age > initialization_parameters[0]:
                selected_trees.append(area_limit_forest_iniPG[i])
                age_exceed_lifetime_index.append(i)
                # area_limit_forest.remove(area_limit_forest[i])
        delete_together(age_exceed_lifetime_index, area_limit_forest_iniPG)
        if len(area_limit_forest_iniPG) > initialization_parameters[4]:
            # 遍历area_limit_forest_iniPG中剩下的树带入求解器（eg knn）算分类准确率，准确率低的放入候选区直至area_limit_forest_iniPG的长度为are_limit的值为止
            num_extra = len(area_limit_forest_iniPG) - initialization_parameters[4]
            print 'num_extra', num_extra
            for i in range(len(area_limit_forest_iniPG)):
                fea_list = numtofea(area_limit_forest_iniPG[i].list, feature)
                if len(fea_list):
                    data_sample = read_data_fea(fea_list, trainX)
                    data_predict = read_data_fea(fea_list, predictX)
                    # acc.append(train_knn(data_sample, trainY, data_predict, predictY))
                    # acc.append(train_svm(data_sample, trainY, data_predict, predictY))#每棵树的准确率存在acc中
                    acc.append(train_tree(data_sample, trainY, data_predict, predictY))
                else:
                    print 'fea_list is null'
                    acc.append(0)
                    # exit(1)

            print 'acc', acc
            print 'acc的长度', len(acc)
            # 将acc中前num_extra的最小值的角标存入acc_omit_index中
            for i in range(num_extra):
                acc_min = min(acc)
                # print('acc_min',acc_min)
                acc_min_index = acc.index(acc_min)
                acc[acc_min_index] = 100
                acc_omit_index.append(acc_min_index)
            print 'acc_omit_indexd的长度', len(acc_omit_index)
            print 'acc_omit_index', acc_omit_index
            # print('max(acc_omit_index）索引的最大值',max(acc_omit_index))
            for each_item in acc_omit_index:
                selected_trees.append(area_limit_forest_iniPG[each_item])
            delete_together(acc_omit_index, area_limit_forest_iniPG)
    return selected_trees


# Global_seeding（全局播种）
def reverse_binary_GSC(vice_verse_attri_GSC, candidate_area, num_fea_original):
    after_reverse = []
    # candidate_area_growing+=candidate_area
    selected_tree_canarea = []  # 从候选区中挑出来进行反转的树
    num_percent_transfer = int(len(candidate_area) * initialization_parameters[3])
    print 'num_percent_transfer', num_percent_transfer
    # 从不断增长的候选区中挑出来进行反转的树
    j = 0
    x = []  # 做测试用 ，可以删除
    while j < num_percent_transfer:
        y = random.randint(0, len(candidate_area) - 1)
        if candidate_area[y] not in selected_tree_canarea:
            selected_tree_canarea.append(candidate_area[y])
            j = j + 1
            x.append(y)
        else:
            continue
    print '从候选区中选出需要进行反转的树的索引值：', x  # 做测试用 ，可以删除
    for i in range(num_percent_transfer):  # 做测试用 ，可以删除
        print '从候选区中选出的需要进行全部反转的树：', selected_tree_canarea[i].list, selected_tree_canarea[i].age
    # 将selected_tree_canarea中每棵树的list转为二进制，长度不够的要补位
    # for i in range(len(selected_tree_canarea)):
    #     str0=bin(selected_tree_canarea[i].list).replace('0b','')
    #     if (len(str0) < num_fea_original):  # 长度不够，开始补位0
    #         j = 0
    #         str1 = ''
    #         short_length = num_fea_original - len(str0)
    #         while j < short_length:
    #             str1 += '0'
    #             j += 1
    #         str0 = str1 + str0
    #     selected_tree_canarea[i].list=str0
    #     print('从候选区中选出的需要进行全部反转的树补位后的二进制形式',selected_tree_canarea[i].list)#做测试用 ，可以删除
    for i in range(len(selected_tree_canarea)):
        temp = Tree(selected_tree_canarea[i].list, selected_tree_canarea[i].age)
        for j in range(len(vice_verse_attri_GSC)):
            if temp.list[vice_verse_attri_GSC[j]] == '0':
                const_value = 1
                new_string = index_replace(vice_verse_attri_GSC[j], temp.list, const_value)
                temp.list = new_string
            else:
                const_value = 0
                new_string = index_replace(vice_verse_attri_GSC[j], temp.list, const_value)
                temp.list = new_string
        # temp.list=int(temp.list,2)
        after_reverse.append(temp)
    # after_reverse
    # selected_tree_canarea[i].list=int(str0,2)
    # for i in range(len(after_reverse)):#做测试用 ，可以删除
    #     print('after_reverse所有属性一起反转后的样子：', after_reverse[i].list, after_reverse[i].age)
    return after_reverse
