#!/usr/bin/python
# -*- coding:utf-8 -*-
import time
import FSFOATOOL as tools
from FSFOA_CORE import *
import util


class Tree:
    def __init__(self, tree_list, tree_age):
        self.list = tree_list
        self.age = tree_age


# 初始化,针对森林中的树构造对应的个体
def ini_reverse(attri_reverse, area_limit_forest_single_tree):
    tree = Tree(area_limit_forest_single_tree.list, area_limit_forest_single_tree.age)
    tree.list = util.revers(tree.list, attri_reverse)
    return tree


# 初始化，构造初始森林
def ini_PG(forest_init):
    area_limit_forest_init = []
    for init in xrange(len(forest_init)):
        '''原始IFSFOA算法,这里不要
        if (i < fs_num_tree_ini):
            attri_reverse = random_form(fs_num_fea)  # 每次选择不同的反转属性
            print('第', i, '棵树选择要反转的属性是：', attri_reverse)
            area_limit_forest_iniPG.append(ini_reverse(attri_reverse, forest_init[i]))
        else:
            attri_reverse = random_form(bs_num_fea)
            print('第', i, '棵树选择要反转的属性是：', attri_reverse)
            area_limit_forest_iniPG.append(ini_reverse(attri_reverse, forest_init[i]))'''
        init_feature_num = np.random.randint(0, num_fea_original)
        attri_reverse = util.random_form(init_feature_num, num_fea_original)
        tree = ini_reverse(attri_reverse, forest_init[init])
        area_limit_forest_init.append(deepcopy(tree))
    return area_limit_forest_init


def FSFOA(area_limit_forest_iniPG):
    original_acc_knn = tools.train_knn(trainX, trainY, predictX, predictY, 3)
    # original_acc_svm = tools.train_svm(trainX, trainY, predictX, predictY)
    # original_acc_tree = tools.train_tree(trainX, trainY, predictX, predictY)
    start = time.clock()
    accuracy_max = []  # 存储循环loop_condition中每次的最大准确率
    accuracy_max_feature = []  # 存储循环loop_condition中每次的最大准确率所对应的特征
    accuracy_max_DR = []  # 存储循环loop_condition中每次的最大准确率所对应的维度缩减
    candidate_area_growing = []  # 存储候选区中的树
    m = 0
    while m < loop_condition:
        print '****************************第', m + 1, '次循环*********************'
        m += 1
        vice_verse_attri = util.random_form(initialization_parameters[1], num_fea_original)
        print '#######################################第', m, 'local seeding播种开始##################################'
        new_tree_nestification = reverse_binary_LSC(vice_verse_attri, area_limit_forest_iniPG)
        for each_item in new_tree_nestification:
            area_limit_forest_iniPG.append(deepcopy(each_item))
        print '#######################################第', m, 'local seeding播种结束##################################'

        print '#######################################第', m, 'population limiting放入候选区开始######################'
        # 获取候选区的树
        candidate_area_growing = select_trees(trainX, trainY, predictX, predictY, initialization_parameters, feature, area_limit_forest_iniPG)
        print '#######################################第', m, 'population limiting 放入候选区结束######################'
        print '#######################################第', m, 'Global seeding GSC开始##################################'
        # TODO 原始论文从候选区随机选择若干棵树全局播种,这里可以改进上启发式
        '''
        # 只需要根据GSC值完成候选区5%的反转即可(这里可以改进)
        
        
        
        '''
        vice_verse_attri_GSC = util.random_form(initialization_parameters[2], num_fea_original)  # 全局播种特征的集合
        after_GSC_reverse = reverse_binary_GSC(initialization_parameters, vice_verse_attri_GSC, candidate_area_growing)
        area_limit_forest_iniPG += after_GSC_reverse
        print '#######################################第', m, 'Global seeding GSC结束##################################'

        print '#######################################第', m, 'update optimal更新最优开始##############################'
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
        print '#######################################第', m, 'update optimal更新最优结束##############################'
    # 循环loop_condition次后的最终全局更新最优值
    accuracy_max_temp = max(accuracy_max)
    accuracy_max_temp_index = accuracy_max.index(accuracy_max_temp)
    for i in range(len(accuracy_max)):
        if (accuracy_max[i] == accuracy_max_temp) and (accuracy_max_DR[i] > accuracy_max_DR[accuracy_max_temp_index]):
            accuracy_max_temp_index = i
        else:
            continue
    optimal_feature_subset = accuracy_max_feature[accuracy_max_temp_index]
    DR = 1 - (1.0 * optimal_feature_subset.count(1) / num_fea_original)
    end = time.clock()

    print '代码运行时间为：', end - start
    print 'feature number of Original data set :  ', num_fea_original  # 原始数据集特征数目
    print 'original_acc_knn is : ', original_acc_knn
    # print 'original_acc_svm is : ', original_acc_svm
    # print 'original_acc_tree is : ', original_acc_tree
    print 'iniPG_optimal_accuracy is : ', max(accuracy_max)
    print 'Optimal Feature subset ：', optimal_feature_subset  # 最优特征子集
    print 'length of candidate area ：', len(candidate_area_growing)  # 候选区的长度
    print 'percent of dimension reduction : ', DR


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


def ADAFSFOA(area_limit_forest_iniPG):
    original_acc_knn = tools.train_knn(trainX, trainY, predictX, predictY, 3)
    # original_acc_svm = tools.train_svm(trainX, trainY, predictX, predictY)
    # original_acc_tree = tools.train_tree(trainX, trainY, predictX, predictY)
    start = time.clock()
    accuracy_max = []  # 存储循环loop_condition中每次的最大准确率
    accuracy_max_feature = []  # 存储循环loop_condition中每次的最大准确率所对应的特征
    accuracy_max_DR = []  # 存储循环loop_condition中每次的最大准确率所对应的维度缩减
    candidate_area_growing = []  # 存储候选区中的树
    m = 0
    while m < loop_condition:
        print '****************************第', m + 1, '次循环*********************'
        m += 1
        vice_verse_attri = util.random_form(initialization_parameters[1], num_fea_original)
        print '#######################################第', m, 'local seeding播种开始##################################'
        new_tree_nestification = reverse_binary_LSC(vice_verse_attri, area_limit_forest_iniPG)
        for each_item in new_tree_nestification:
            area_limit_forest_iniPG.append(deepcopy(each_item))
        print '#######################################第', m, 'local seeding播种结束##################################'

        print '#######################################第', m, 'population limiting放入候选区开始######################'
        # 获取候选区的树
        candidate_area_growing = select_trees(trainX, trainY, predictX, predictY, initialization_parameters, feature,
                                              area_limit_forest_iniPG)
        print '#######################################第', m, 'population limiting 放入候选区结束######################'
        print '#######################################第', m, 'Global seeding GSC开始##################################'
        # TODO 原始论文从候选区随机选择若干棵树全局播种,这里可以改进上启发式
        '''
        # 只需要根据GSC值完成候选区5%的反转即可(这里可以改进)



        '''
        vice_verse_attri_GSC = util.random_form(initialization_parameters[2], num_fea_original)  # 全局播种特征的集合
        after_GSC_reverse = reverse_binary_GSC(initialization_parameters, vice_verse_attri_GSC, candidate_area_growing)
        area_limit_forest_iniPG += after_GSC_reverse
        print '#######################################第', m, 'Global seeding GSC结束##################################'

        print '#######################################第', m, 'update optimal更新最优开始##############################'
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
        print '#######################################第', m, 'update optimal更新最优结束##############################'
    # 循环loop_condition次后的最终全局更新最优值
    accuracy_max_temp = max(accuracy_max)
    accuracy_max_temp_index = accuracy_max.index(accuracy_max_temp)
    for i in range(len(accuracy_max)):
        if (accuracy_max[i] == accuracy_max_temp) and (accuracy_max_DR[i] > accuracy_max_DR[accuracy_max_temp_index]):
            accuracy_max_temp_index = i
        else:
            continue
    optimal_feature_subset = accuracy_max_feature[accuracy_max_temp_index]
    DR = 1 - (1.0 * optimal_feature_subset.count(1) / num_fea_original)
    end = time.clock()

    print '代码运行时间为：', end - start
    print 'feature number of Original data set :  ', num_fea_original  # 原始数据集特征数目
    print 'original_acc_knn is : ', original_acc_knn
    # print 'original_acc_svm is : ', original_acc_svm
    # print 'original_acc_tree is : ', original_acc_tree
    print 'iniPG_optimal_accuracy is : ', max(accuracy_max)
    print 'Optimal Feature subset ：', optimal_feature_subset  # 最优特征子集
    print 'length of candidate area ：', len(candidate_area_growing)  # 候选区的长度
    print 'percent of dimension reduction : ', DR


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
    inputDict = {'ionosphere': 'ionosphere', 'cleveland': 'cleveland', 'wine': 'wine', 'sonar': 'sonar',
                 'srbct': 'srbct',
                 'segmentation': 'segmentation', 'vehicle': 'vehicle', 'dermatology': 'dermatology', 'heart': 'heart',
                 'glass': 'glass', 'arcene': 'arcene'}
    # trainX,trainY,predictX,predictY are all list
    trainX, trainY, predictX, predictY, loop_condition, initialization_parameters = util.loadData(inputDict['segmentation'])
    # TODO
    num_tree_ini = 50  # 初始化时森林中tree的个数 ， 这里可以改进
    initial_forest = []
    area_limit_forest = []
    num_fea_original = np.mat(trainX).shape[1]  # 特征长度
    feature = []  # 特征集合索引,特征集合的角标
    for i in range(num_fea_original):
        feature.append(i)
    # TODO 初始化策略(这里上启发式),可以上决策树熵理论，不随机播特征，播数据（或是取子集kmeans++之后播）
    '''





    '''

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
    area_limit_forest = [deepcopy(Tree(initial_forest, 0)) for row in xrange(num_tree_ini)]
    area_limit_forest_iniPG = ini_PG(area_limit_forest)
    FSFOA_iniPG = deepcopy(area_limit_forest_iniPG)
    ADAFSFOA_iniPG = deepcopy(area_limit_forest_iniPG)
    # for i in xrange(len(area_limit_forest_iniPG)):
    #     print '初始化树', area_limit_forest_iniPG[i].list, area_limit_forest_iniPG[i].age
    FSFOA(FSFOA_iniPG)
    ADAFSFOA(ADAFSFOA_iniPG)