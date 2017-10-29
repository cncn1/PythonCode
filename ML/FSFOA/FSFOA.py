#!/usr/bin/python
# -*- coding:utf-8 -*-
import time
import util
from FSFOA_CORE import *
import ADAFSFOA_CORE as ada


# 特征树
class Tree:
    def __init__(self, tree_list, tree_age):
        self.list = tree_list
        self.age = tree_age


# 初始化,针对森林中的树构造对应的个体
def ini_reverse(attri_reverse, area_limit_forest_single_tree):
    tree = Tree(area_limit_forest_single_tree.list, area_limit_forest_single_tree.age)
    tree.list = revers(tree.list, attri_reverse)
    return tree


# 初始化，构造初始森林
def ini_PG(forest_init, optimalFeature=None):
    area_limit_forest_init = []
    for init in xrange(len(forest_init)):
        init_feature_num = random.randint(0, num_fea_original)
        attri_reverse = random_form(init_feature_num, num_fea_original)
        tree = ini_reverse(attri_reverse, forest_init[init])
        area_limit_forest_init.append(deepcopy(tree))
    if optimalFeature is None:
        return area_limit_forest_init
    else:
        for adaTree in area_limit_forest_init:
            adaTree.list[optimalFeature] = 1
        return area_limit_forest_init


def FSFOA(area_limit_forest_iniPG):
    # original_acc_knn = train_knn(trainX, trainY, predictX, predictY, 3)
    # original_acc_svm = train_svm(trainX, trainY, predictX, predictY)
    # original_acc_tree = train_tree(trainX, trainY, predictX, predictY)
    start = time.clock()
    accuracy_max = []  # 存储循环loop_condition中每次的最大准确率
    accuracy_max_feature = []  # 存储循环loop_condition中每次的最大准确率所对应的特征
    accuracy_max_DR = []  # 存储循环loop_condition中每次的最大准确率所对应的维度缩减
    candidate_area_growing = []  # 存储候选区中的树
    m = 0
    while m < loop_condition:
        # print '****************************第', m + 1, '次循环*********************'
        m += 1
        vice_verse_attri = random_form(initialization_parameters[1], num_fea_original)
        # print '#######################################第', m, 'local seeding播种开始##################################'
        new_tree_nestification = reverse_binary_LSC(vice_verse_attri, area_limit_forest_iniPG)
        for each_item in new_tree_nestification:
            area_limit_forest_iniPG.append(deepcopy(each_item))
        # print '#######################################第', m, 'local seeding播种结束##################################'

        # print '#######################################第', m, 'population limiting放入候选区开始######################'
        # 获取候选区的树
        candidate_area_growing = select_trees(trainX, trainY, predictX, predictY, initialization_parameters[4], feature,
                                              area_limit_forest_iniPG)
        # print '#######################################第', m, 'population limiting 放入候选区结束######################'
        # print '#######################################第', m, 'Global seeding GSC开始##################################'
        # TODO 原始论文从候选区随机选择若干棵树全局播种,这里可以改进上启发式
        '''
        # 只需要根据GSC值完成候选区5%的反转即可(这里可以改进)
        
        
        
        '''
        vice_verse_attri_GSC = random_form(initialization_parameters[2], num_fea_original)  # 全局播种特征的集合
        after_GSC_reverse = reverse_binary_GSC(initialization_parameters[3], vice_verse_attri_GSC, candidate_area_growing)
        area_limit_forest_iniPG += after_GSC_reverse
        # print '#######################################第', m, 'Global seeding GSC结束##################################'

        # print '#######################################第', m, 'update optimal更新最优开始##############################'
        acc = []
        DR = []
        for tree_update in area_limit_forest_iniPG:
            fea_list = numtofea(tree_update.list, feature)
            if len(fea_list):
                data_sample = read_data_fea(fea_list, trainX)
                data_predict = read_data_fea(fea_list, predictX)
                acc.append(train_knn(data_sample, trainY, data_predict, predictY))  # 每棵树的准确率存在acc中
                # acc.append(train_svm(data_sample, trainY, data_predict, predictY))
                # acc.append(train_tree(data_sample, trainY, data_predict, predictY))
                DR.append(1 - (1.0 * len(fea_list) / len(feature)))
            else:
                acc.append(0)
                DR.append(0)
        acc_max = max(acc)
        acc_max_index = acc.index(acc_max)
        for i in xrange(len(acc)):
            if (acc[i] == acc_max) and (DR[i] > DR[acc_max_index]):
                acc_max_index = i
            else:
                continue
        accuracy_max.append(acc_max)
        tree_max = area_limit_forest_iniPG[acc_max_index]  # 找到最优树在area_limit_forest_iniPG中的位置,最优树即准确率最高的特征子集。
        accuracy_max_feature.append(tree_max.list)
        accuracy_max_DR.append(DR[acc_max_index])
        area_limit_forest_iniPG[acc_max_index].age = 0  # 最优树的age设为0
        # print '#######################################第', m, 'update optimal更新最优结束##############################'
    # 循环loop_condition次后的最终全局更新最优值
    accuracy_max_temp = max(accuracy_max)
    accuracy_max_temp_index = accuracy_max.index(accuracy_max_temp)
    for i in range(len(accuracy_max)):
        if (accuracy_max[i] == accuracy_max_temp) and (accuracy_max_DR[i] > accuracy_max_DR[accuracy_max_temp_index]):
            accuracy_max_temp_index = i
        else:
            continue
    optimal_feature_subset = accuracy_max_feature[accuracy_max_temp_index]
    accuracy = max(accuracy_max)
    DR = 1 - (1.0 * optimal_feature_subset.count(1) / num_fea_original)
    end = time.clock()

    # print '代码运行时间为：', end - start
    # print 'feature number of Original data set :  ', num_fea_original  # 原始数据集特征数目
    # print 'original_acc_knn is : ', original_acc_knn
    # print 'original_acc_svm is : ', original_acc_svm
    # print 'original_acc_tree is : ', original_acc_tree
    # print 'Optimal Feature subset ：', optimal_feature_subset  # 最优特征子集
    # print 'length of candidate area ：', len(candidate_area_growing)  # 候选区的长度
    print 'FSFOA_accuracy is : ', accuracy, '\tFSFOA_DR is : ', DR, '\n'
    return accuracy, DR


    # print '#########将改进后的算法所算出的特征子集字符串直接带入分类器，得到相同的准确率即实验具备可重复性。#########'
    # fea_list_CB = numtofea(optimal_feature_subset, feature)
    # print 'fea_list_CB:', fea_list_CB
    # data_sample = read_data_fea(fea_list_CB, trainX)
    # data_predict = read_data_fea(fea_list_CB, predictX)
    # zhunquelvKNN = train_knn(data_sample, trainY, data_predict, predictY)
    # zhunquelvSVM = train_svm(data_sample, trainY, data_predict, predictY)
    # zhunquelvTREE = train_tree(data_sample, trainY, data_predict, predictY)
    # weidu = 1 - (1.0 * len(fea_list_CB) / len(feature))
    # print 'KNN准确率:', zhunquelvKNN
    # print 'SVM准确率:', zhunquelvSVM
    # print 'J48准确率:', zhunquelvTREE
    # print '维度缩减：', weidu


# optimalFeature 为根据信息熵挑出的最优特征
def ADAFSFOA(area_limit_forest_iniPG):
    # original_acc_knn = train_knn(trainX, trainY, predictX, predictY, 3)
    # original_acc_svm = train_svm(trainX, trainY, predictX, predictY)
    # original_acc_tree = train_tree(trainX, trainY, predictX, predictY)
    start = time.clock()
    accuracy_max = []  # 存储循环loop_condition中每次的最大准确率
    accuracy_max_feature = []  # 存储循环loop_condition中每次的最大准确率所对应的特征
    accuracy_max_DR = []  # 存储循环loop_condition中每次的最大准确率所对应的维度缩减
    GSC = min((2 + 2 * loop_condition), int(num_fea_original * 0.5))  # 用模拟退火启发式的方法，可以让GSC有更广泛的搜索空间
    m = 0
    while m < loop_condition:
        # print '****************************第', m + 1, '次循环*********************'
        m += 1
        vice_verse_attri = random_form(initialization_parameters[1], num_fea_original)
        # print '#######################################第', m, 'local seeding播种开始##################################'
        new_tree_nestification = reverse_binary_LSC(vice_verse_attri, area_limit_forest_iniPG)
        for each_item in new_tree_nestification:
            area_limit_forest_iniPG.append(deepcopy(each_item))
        # print '#######################################第', m, 'local seeding播种结束##################################'

        # print '#######################################第', m, 'population limiting放入候选区开始######################'
        # 获取候选区的树
        candidate_area_growing = select_trees(trainX, trainY, predictX, predictY, initialization_parameters[0], initialization_parameters[4], feature, area_limit_forest_iniPG)
        # print '#######################################第', m, 'population limiting 放入候选区结束#####################'
        # print '#######################################第', m, 'Global seeding GSC开始#################################'
        # TODO 原始论文从候选区随机选择若干棵树全局播种,这里可以改进上启发式
        '''
        # 只需要根据GSC值完成候选区5%的反转即可(这里可以改进)



        '''
        # GSC受退火函数启发
        GSC = ada.T(GSC, m)
        vice_verse_attri_GSC = random_form(GSC, num_fea_original)  # 全局播种特征的集合
        # initialization_parameters[3] 是转化率，这个为什么不需要上启发式函数，个人认为是因为候选区中的树会越来越多
        after_GSC_reverse = reverse_binary_GSC(initialization_parameters[3], vice_verse_attri_GSC, candidate_area_growing)
        area_limit_forest_iniPG += after_GSC_reverse
        # print '#######################################第', m, 'Global seeding GSC结束##################################'

        # print '#######################################第', m, 'update optimal更新最优开始##############################'
        acc = []
        DR = []
        for tree_update in area_limit_forest_iniPG:
            fea_list = numtofea(tree_update.list, feature)
            if len(fea_list):
                data_sample = read_data_fea(fea_list, trainX)
                data_predict = read_data_fea(fea_list, predictX)
                # acc.append(train_knn(data_sample, trainY, data_predict, predictY))  # 每棵树的准确率存在acc中
                # acc.append(train_svm(data_sample, trainY, data_predict, predictY))
                acc.append(train_tree(data_sample, trainY, data_predict, predictY))
                DR.append(1 - (1.0 * len(fea_list) / len(feature)))
            else:
                acc.append(0)
                DR.append(0)
        acc_max = max(acc)
        acc_max_index = acc.index(acc_max)
        for i in xrange(len(acc)):
            if (acc[i] == acc_max) and (DR[i] > DR[acc_max_index]):
                acc_max_index = i
            else:
                continue
        accuracy_max.append(acc_max)
        tree_max = area_limit_forest_iniPG[acc_max_index]  # 找到最优树在area_limit_forest_iniPG中的位置,最优树即准确率最高的特征子集。
        accuracy_max_feature.append(tree_max.list)
        accuracy_max_DR.append(DR[acc_max_index])
        area_limit_forest_iniPG[acc_max_index].age = 0  # 最优树的age设为0
        # print '#######################################第', m, 'update optimal更新最优结束##############################'
    # 循环loop_condition次后的最终全局更新最优值
    accuracy_max_temp = max(accuracy_max)
    accuracy_max_temp_index = accuracy_max.index(accuracy_max_temp)
    for i in xrange(len(accuracy_max)):
        if (accuracy_max[i] == accuracy_max_temp) and (accuracy_max_DR[i] > accuracy_max_DR[accuracy_max_temp_index]):
            accuracy_max_temp_index = i
        else:
            continue
    optimal_feature_subset = accuracy_max_feature[accuracy_max_temp_index]  # 结束迭代后产生的最优子集
    # 群体选优策略，一个是可能出现最优准率的特征子集，一个是可能进一步维度缩减的特征子集
    last_compare_subset_accuracy, last_compare_subset_DR = ada.GroupSelection(area_limit_forest_iniPG, num_fea_original, optimal_feature_subset.count(1))
    accuracy = max(accuracy_max)
    DR = 1 - (1.0 * optimal_feature_subset.count(1) / num_fea_original)
    end = time.clock()

    # print '代码运行时间为：', end - start
    # print 'feature number of Original data set :  ', num_fea_original  # 原始数据集特征数目
    # print 'original_acc_knn is : ', original_acc_knn
    # print 'original_acc_svm is : ', original_acc_svm
    # print 'original_acc_tree is : ', original_acc_tree
    # print 'Optimal Feature subset ：', optimal_feature_subset  # 最优特征子集
    # print 'length of candidate area ：', len(candidate_area_growing)  # 候选区的长度
    print 'ADAFSFOA_accuracy is : ', accuracy, '\tADAFSFOA_DR is : ', DR, '\n'
    return accuracy, DR

    # print '#########将改进后的算法所算出的特征子集字符串直接带入分类器，得到相同的准确率即实验具备可重复性。****#***'
    # fea_list_CB = numtofea(optimal_feature_subset, feature)
    # print 'fea_list_CB:', fea_list_CB
    # data_sample = read_data_fea(fea_list_CB, trainX)
    # data_predict = read_data_fea(fea_list_CB, predictX)
    # zhunquelvKNN = train_knn(data_sample, trainY, data_predict, predictY)
    # zhunquelvSVM = train_svm(data_sample, trainY, data_predict, predictY)
    # zhunquelvTREE = train_tree(data_sample, trainY, data_predict, predictY)
    # weidu = 1 - (1.0 * len(fea_list_CB) / len(feature))
    # print 'KNN准确率:', zhunquelvKNN
    # print 'SVM准确率:', zhunquelvSVM
    # print 'J48准确率:', zhunquelvTREE
    # print '维度缩减：', weidu


if __name__ == '__main__':
    # 变量定义
    inputDict = {'ionosphere': ['ionosphere', [1, 1, 10, 2, 1, 2, 37, 1, 10]], 'cleveland': ['cleveland', [37, 1, 1]],
                 'wine': ['wine', [1, 1, 10, 2, 1, 2, 37, 1, 9]], 'sonar': ['sonar', [1, 1, 10, 2, 1, 2, 37, 1, 10]],
                 'srbct': ['srbct', [37, 1, 10]], 'segmentation': ['segmentation', [1, 1, 10]],
                 'vehicle': ['vehicle', [1, 1, 10, 2, 1, 2, 37, 1, 1]],
                 'dermatology': ['dermatology', [1, 1, 10, 37, 1, 10]], 'heart': ['heart', [1, 1, 10, 2, 1, 2]],
                 'glass': ['glass', [1, 1, 10, 2, 1, 2, 37, 1, 1]], 'arcene': ['arcene', [1, 1, 1]]}
    for key in inputDict:
        dataSet = inputDict['dermatology']
        loop0 = len(dataSet[1]) / 3  # 实验组数
        for loop in xrange(loop0):
            labName = dataSet[1][(loop * 3)]  # 每组实验具体内容
            labTimes = dataSet[1][(loop * 3) + 1]  # 每组实验重复次数
            fileNum = dataSet[1][(loop * 3 + 2)]  # 每组实验文件个数
            FSFOA_accuracy_total = 0  # 记录FSFOA算法准确率之和
            FSFOA_DR_total = 0  # 记录FSFOA算法DR之和
            ADAFSFOA_accuracy_total = 0  # 记录ADAFSFOA算法准确率之和
            ADAFSFOA_DR_total = 0  # 记录ADAFSFOA算法DR之和
            count = 0  # 记录总的实验次数
            for times in xrange(labTimes):
                for eachfile in xrange(fileNum):
                    count += 1
                    # trainX,trainY,predictX,predictY are all list
                    trainX, trainY, predictX, predictY, loop_condition, initialization_parameters = util.loadData(dataSet[0], labName, eachfile + 1)
                    # TODO
                    num_tree_ini = 50  # 初始化时森林中tree的个数 ， 这里可以改进
                    initial_forest = []
                    area_limit_forest = []
                    num_fea_original = mat(trainX).shape[1]  # 特征长度
                    feature = []  # 特征集合索引,特征集合的角标
                    for i in range(num_fea_original):
                        feature.append(i)
                    # 将每棵树记录的特征以全0数组的形式初始化
                    initial_forest = [0] * num_fea_original
                    # 初始化森林
                    area_limit_forest = [deepcopy(Tree(initial_forest, 0)) for row in xrange(num_tree_ini)]
                    # FSFOA_iniPG = ini_PG(area_limit_forest)
                    # FSFOA_accuracy, FSFOA_DR = FSFOA(FSFOA_iniPG)
                    # FSFOA_accuracy_total += FSFOA_accuracy
                    # FSFOA_DR_total += FSFOA_DR
                    # TODO 初始化策略(这里上启发式),可以上决策树熵理论，不随机播特征，播数据（或是取子集kmeans++之后播）
                    '''





                    '''
                    # 改进一：根据信息熵理论，挑出具有最好用于划分数据集的特征
                    optimalFeature = ada.chooseBestFeatureToSplit(trainX)
                    ADAFSFOA_iniPG = ini_PG(area_limit_forest, optimalFeature=optimalFeature)
                    ADAFSFOA_accuracy, ADAFSFOA_DR = ADAFSFOA(ADAFSFOA_iniPG)
                    ADAFSFOA_accuracy_total += ADAFSFOA_accuracy
                    ADAFSFOA_DR_total += ADAFSFOA_DR

            # FSFOA_accuracy_mean = FSFOA_accuracy_total / count
            # FSFOA_DR_mean = FSFOA_DR_total / count
            # util.print_to_file('FSFOA', dataSet[0], labName, FSFOA_accuracy_mean * 100, FSFOA_DR_mean * 100)

            ADAFSFOA_accuracy_mean = ADAFSFOA_accuracy_total / count
            ADAFSFOA_DR_mean = ADAFSFOA_DR_total / count
            # util.print_to_file('ADAFSFOA', dataSet[0], labName, ADAFSFOA_accuracy_mean * 100, ADAFSFOA_DR_mean * 100)
