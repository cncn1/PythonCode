#!/usr/bin/python
# -*- coding:utf-8 -*-
import time
from FSFOA_CORE import *
from EFSFOA_CORE import *


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
def ini_PG(forest_init):
    area_limit_forest_init = []
    for init in xrange(len(forest_init)):
        init_feature_num = random.randint(0, num_fea_original)
        attri_reverse = random_form(init_feature_num, num_fea_original)
        tree = ini_reverse(attri_reverse, forest_init[init])
        area_limit_forest_init.append(deepcopy(tree))
    for adaTree in area_limit_forest_init[0: int(len(area_limit_forest_init) * 0.5)]:
            adaTree.list = revers(adaTree.list, optimalFeature, 0)
    return area_limit_forest_init


# optimalFeature 为根据信息熵挑出的最优特征
def EFSFOA(area_limit_forest_iniPG):
    start = time.clock()
    accuracy_max = []  # 存储循环loop_condition中每次的最大准确率
    accuracy_max_feature = []  # 存储循环loop_condition中每次的最大准确率所对应的特征
    accuracy_max_DR = []  # 存储循环loop_condition中每次的最大准确率所对应的维度缩减
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
        # 更新区域中的树，同时获取候选区的树
        candidate_area_growing = select_trees(trainX, trainY, predictX, predictY, initialization_parameters[0],
                                              initialization_parameters[4], feature, area_limit_forest_iniPG,
                                              trainSelect, KinKNN)
        # print '#######################################第', m, 'population limiting 放入候选区结束#####################'
        # print '#######################################第', m, 'Global seeding GSC开始#################################'
        # TODO 原始论文从候选区随机选择若干棵树全局播种,这里可以改进上启发式
        '''
        # 只需要根据转化率值完成候选区5%的反转即可(这里可以改进)
        '''
        # initialization_parameters[3] 是转化率，这个为什么不需要上启发式函数，个人认为是因为候选区中的树会越来越多已经达到了动态的变化过程
        GSC0 = 1.0 * min(2 + 2 * initialization_parameters[0], num_fea_original * 0.5)  # GSC的初值设置
        after_GSC_reverse = reverse_binary_GSC_EFSFOA(initialization_parameters[3], candidate_area_growing, num_fea_original,
                                               GSC0)
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
                acc.append(trainSelect(data_sample, trainY, data_predict, predictY, KinKNN))  # 每棵树的准确率存在acc中
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
    last_compare_subset_accuracy, last_compare_subset_DR = GroupSelection(area_limit_forest_iniPG, num_fea_original,
                                                                          optimal_feature_subset.count('1'))
    if last_compare_subset_DR.count('1') == 0:
        resultList = [optimal_feature_subset, last_compare_subset_accuracy]
    else:
        resultList = [optimal_feature_subset, last_compare_subset_accuracy, last_compare_subset_DR]
    accuracy, DR, optimal_subset = OptimalResult(trainX, trainY, predictX, predictY, resultList, feature,
                                                 loop_condition,
                                                 trainSelect, KinKNN)
    # accuracy = max(accuracy_max)
    # DR = 1 - (1.0 * optimal_feature_subset.count(1) / num_fea_original)
    end = time.clock()

    print '代码运行时间为: ', end - start
    # print 'original_acc is : ', original_acc
    print 'Optimal Feature subset: ', optimal_feature_subset  # 最优特征子集
    # print 'length of candidate area ：', len(candidate_area_growing)  # 候选区的长度
    print 'EFSFOA_accuracy is: ', accuracy, '\tEFSFOA_DR is: ', DR, '\n'
    return accuracy, DR, optimal_feature_subset, end - start


if __name__ == '__main__':
    # 变量定义
    inputDict0 = {'ionosphere': ['ionosphere', [1, 1, 10, 2, 5, 2, 37, 1, 10]], 'cleveland': ['cleveland', [37, 1, 1]],
                  'wine': ['wine', [1, 1, 10, 2, 5, 2, 37, 1, 9]], 'sonar': ['sonar', [1, 1, 10, 2, 5, 2, 37, 1, 10]],
                  'segmentation': ['segmentation', [1, 1, 10]], 'vehicle': ['vehicle', [1, 1, 10, 2, 5, 2, 37, 1, 1]],
                  'dermatology': ['dermatology', [1, 1, 10, 37, 1, 10]], 'heart': ['heart', [1, 1, 10, 2, 5, 2]],
                  'glass': ['glass', [1, 1, 10, 2, 5, 2, 37, 1, 1]], 'z1': ['srbct', [37, 1, 10]]}
    inputDict1 = {'z2': ['arcene', [37, 1, 1]]}
    # trainName = ['J48', 'SVM', '1NN', '3NN', '5NN']
    DataSets, trainName = inputDict1, '5NN',
    print trainName, '\n'
    trainSelect, KinKNN = select_train(trainName)  # 选择分类器
    for key in DataSets:
        dataSet = DataSets[key]
        loop0 = len(dataSet[1]) / 3  # 实验组数
        for loop in xrange(loop0):
            labName = dataSet[1][(loop * 3)]  # 每组实验具体内容
            labTimes = dataSet[1][(loop * 3) + 1]  # 每组实验重复次数
            fileNum = dataSet[1][(loop * 3 + 2)]  # 每组实验文件个数
            EFSFOA_accuracy_total = 0  # 记录EFSFOA算法准确率之和
            EFSFOA_DR_total = 0  # 记录EFSFOA算法DR之和
            EFSFOA_OPSUBALL = []  # 记录EFSFOA算法选出的最优特征子集的集合
            EFSFOA_TIMEALL = 0  # 记录EFSFOA算法运行的总时间
            count = 0  # 记录总的实验次数
            for times in xrange(labTimes):
                for eachfile in xrange(fileNum):
                    count += 1
                    # trainX,trainY,predictX,predictY are all list
                    trainX, trainY, predictX, predictY, loop_condition, initialization_parameters = util.loadData(
                        dataSet[0], labName, eachfile + 1)
                    # TODO
                    num_tree_ini = 50  # 初始化时森林中tree的个数, 这里可以改进
                    initial_forest = []
                    area_limit_forest = []
                    num_fea_original = mat(trainX).shape[1]  # 特征长度
                    feature = []  # 特征集合索引,特征集合的角标
                    for i in range(num_fea_original):
                        feature.append(i)
                    # 将每棵树记录的特征以全0的形式初始化
                    initial_forest = '0' * num_fea_original
                    # 初始化森林
                    area_limit_forest = [deepcopy(Tree(initial_forest, 0)) for row in xrange(num_tree_ini)]
                    # TODO 初始化策略(这里上启发式),可以上决策树熵理论，不随机播特征，播数据（或是取子集kmeans++之后播）
                    '''
                    '''
                    # 改进一：根据信息增益，挑出具有最好用于划分数据集的特征，后续转成根据信息增益比启发50%
                    optimalFeature = chooseBestFeatureToSplit(trainX)
                    EFSFOA_iniPG = ini_PG(area_limit_forest)
                    EFSFOA_accuracy, EFSFOA_DR, EFSFOA_OPSUB, EFSFOA_TIME = EFSFOA(EFSFOA_iniPG)
                    EFSFOA_accuracy_total += EFSFOA_accuracy
                    EFSFOA_DR_total += EFSFOA_DR
                    EFSFOA_OPSUBALL.append(EFSFOA_OPSUB)
                    EFSFOA_TIMEALL += EFSFOA_TIME
            EFSFOA_accuracy_mean = EFSFOA_accuracy_total / count
            EFSFOA_DR_mean = EFSFOA_DR_total / count
            util.print_to_file('EFSFOA', trainName, dataSet[0], labName, EFSFOA_accuracy_mean * 100,
                               EFSFOA_DR_mean * 100,
                               EFSFOA_OPSUBALL, EFSFOA_TIMEALL)
