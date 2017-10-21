import time
from FSFOATOOL import *
from numpy import *
from  FSFOA_CORE import *
from PSO_initial import *
import itertools  # 处理list的嵌套问题
import \
    copy  # 用于深度拷贝列表eg area_limit_forest_age0=copy.deepcopy(area_limit_forest),area_limit_forest_age0的变化不会影响area_limit_forest的变化。

# print('feature index', feature)
# print('原数据集特征数目：',num_fea_original)
# original_acc=train_knn(trainX, trainy,predictX, predicty)
# original_acc=train_svm(trainX, trainy,predictX, predicty)
original_acc = train_tree(trainX, trainy, predictX, predicty)
start = time.clock()
accuracy_max = []  # 存储循环loop_condition中每次的最大准确率
accuracy_max_feature = []  # 存储循环loop_condition中每次的最大准确率所对应的特征
accuracy_max_DR = []  # 存储循环loop_condition中每次的最大准确率所对应的维度缩减

candidate_area = []
candidate_area_growing = []
candidate_area_temp = []
m = 0
while (m < loop_condition):
    print('￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥第', m + 1, '次循环￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥')
    m += 1
    vice_verse_attri = []
    j = 0
    while j < initialization_parameters[1]:
        y = random.randint(0, num_fea_original - 1)
        if y not in vice_verse_attri:
            vice_verse_attri.append(y)
            j = j + 1
        else:
            continue
    print('局部播种时选出的需要反转属性', vice_verse_attri)
    print('增加新树前area_limit_forest_iniPG的长度', len(area_limit_forest_iniPG))

    print(
    '########################################第', m, 'local seeding播种开始###########################################')
    new_tree_nestification = []  # reverse_binary函数调用后返回的list会有嵌套
    new_tree_nonestification = []
    new_tree_nestification.append(reverse_binary(vice_verse_attri, area_limit_forest_iniPG))
    new_tree_nonestification = list(itertools.chain.from_iterable(new_tree_nestification))
    print('new_tree_nonestification增长出新树的长度', len(new_tree_nonestification))
    for i in range(len(new_tree_nonestification)):
        print('新生成的树', new_tree_nonestification[i].list, new_tree_nonestification[i].age)
    for i in range(len(area_limit_forest_iniPG)):
        print('插入新树之前area_limit_forest_iniPG的样子', area_limit_forest_iniPG[i].list, area_limit_forest_iniPG[i].age)
    # 向area_limit_forest_iniPG里插入新树
    for each_item in new_tree_nonestification:
        area_limit_forest_iniPG.append(each_item)
    print('加入新树后area_limit_forest_iniPG的长度', len(area_limit_forest_iniPG))
    for i in range(len(area_limit_forest_iniPG)):
        print('area_limit_forest_iniPG', area_limit_forest_iniPG[i].list, area_limit_forest_iniPG[i].age)

    print('########################################第', m, 'local seeding播种结束######################################')

    print('#######################################第', m, 'population limiting 放入候选区开始###############################')
    candidate_area_growing.append(select_trees(area_limit_forest_iniPG))
    # new_tree_nonestification = list(itertools.chain.from_iterable(new_tree_nestification))
    candidate_area_temp = list(itertools.chain.from_iterable(candidate_area_growing))
    # for i in range(len(candidate_area_temp)):
    #     if isinstance(candidate_area_temp[i].list,str):
    #         candidate_area_temp[i].list=int(candidate_area_temp[i].list,2)
    #     else:
    #         continue
    candidate_area = candidate_area_temp
    print('候选区candidate_area的长度：', len(candidate_area))
    for i in range(len(candidate_area)):
        print('候选区candidate_area中准确率最小前num_extra颗树的list值，age值：', candidate_area[i].list, candidate_area[i].age)
    print('移除多余树area_limit_forest_iniPG的长度', len(area_limit_forest_iniPG))
    for i in range(len(area_limit_forest_iniPG)):
        print(
        '移除多余树area_limit_forest_iniPG的list值，age值：', area_limit_forest_iniPG[i].list, area_limit_forest_iniPG[i].age)
    print('#######################################第', m, 'population limiting 放入候选区结束#################################')

    print(
    '#######################################第', m, 'Global seeding GSC开始#############################################')
    # 只需要根据GSC值完成候选区5%的反转即可
    vice_verse_attri_GSC = []
    j = 0
    while j < initialization_parameters[2]:
        y = random.randint(0, num_fea_original - 1)
        if y not in vice_verse_attri_GSC:
            vice_verse_attri_GSC.append(y)
            j = j + 1
        else:
            continue
    print('全局播种时选出的需要反转属性', vice_verse_attri_GSC)
    after_GSC_reverse = []  # 存放经过reverse_binary_GSC反转后的结果
    after_GSC_reverse = reverse_binary_GSC(vice_verse_attri_GSC, candidate_area, num_fea_original)
    for i in range(len(after_GSC_reverse)):  # 做测试用 ，可以删除
        print('所有属性一起反转后的样子：', after_GSC_reverse[i].list, after_GSC_reverse[i].age)
    area_limit_forest_iniPG += after_GSC_reverse
    print('加入全局播种后的area_limit_forest_iniPG的长度：', len(area_limit_forest_iniPG))
    for i in range(len(area_limit_forest_iniPG)):  # 做测试用 ，可以删除
        print('加入全局播种后的area_limit_forest_iniPG：', area_limit_forest_iniPG[i].list, area_limit_forest_iniPG[i].age)
    print('#######################################第', m,
          'Global seeding GSC结束################################################')

    print(
    '#######################################第', m, 'update optimal更新最优开始#############################################')
    print('updating....')
    acc = []
    DR = []
    for i in range(len(area_limit_forest_iniPG)):
        fea_list = numtofea(area_limit_forest_iniPG[i].list, feature)
        if len(fea_list):
            data_sample = read_data_fea(fea_list, trainX)
            data_predict = read_data_fea(fea_list, predictX)
            # acc.append(train_knn(data_sample, trainy, data_predict, predicty))  # 每棵树的准确率存在acc中
            # acc.append(train_svm(data_sample, trainy, data_predict, predicty))
            acc.append(train_tree(data_sample, trainy, data_predict, predicty))
            DR.append(1 - (len(fea_list) / len(feature)))

        else:
            acc.append(0)
            DR.append(0)
    acc_max = max(acc)
    acc_max_index = acc.index(acc_max)
    for i in range(len(acc)):
        if (acc[i] == acc_max) and (DR[i] > DR[acc_max_index]):
            acc_max_index = i
        else:
            continue
    accuracy_max.append(acc_max)
    tree_max = area_limit_forest_iniPG[acc_max_index]  # 找到最优树在area_limit_forest_iniPG中的位置,最优树即准确率最高的特征子集。
    accuracy_max_feature.append(tree_max.list)
    accuracy_max_DR.append(DR[acc_max_index])
    area_limit_forest_iniPG[acc_max_index].age = 0  # 最优树的age设为0

    print(
    '#######################################第', m, 'update optimal更新最优结束##########################################')
# 循环loop_condition次后的最终全局更新最优值
accuracy_max_temp = max(accuracy_max)
accuracy_max_temp_index = accuracy_max.index(accuracy_max_temp)
for i in range(len(accuracy_max)):
    if (accuracy_max[i] == accuracy_max_temp) and (accuracy_max_DR[i] > accuracy_max_DR[accuracy_max_temp_index]):
        accuracy_max_temp_index = i
    else:
        continue

end = time.clock()

print('代码运行时间为：', end - start)
print('feature index : ', feature)
print('feature number of Original data set :  ', num_fea_original)  # 原始数据集特征数目
print('original_accuracy is : ', original_acc)
print('iniPG_optimal_accuracy is : ', max(accuracy_max))
print('Feature subset ：', accuracy_max_feature[accuracy_max_temp_index])  # 特征子集
print('length of candidate area ：', len(candidate_area))  # 候选区的长度
print('percent of dimension reduction : ', accuracy_max_DR[accuracy_max_temp_index])













# 处理dimension reduction
# num_one_index=numtofea(accuracy_max_feature[accuracy_max.index(max(accuracy_max))],feature)#被选中的最优特征子集中‘1’的索引，以列表的形式返回
# num_one=len(num_one_index)
# DR=1-(num_one/len(feature))


# print('将改进后的算法所算出的特征子集字符串直接带入分类器，得到相同的准确率即实验具备可重复性。*********************************************')
#
# fea_list_CB = numtofea('0100011011100101000',feature)
# print('fea_list_CB:',fea_list_CB)
# data_sample = read_data_fea(fea_list_CB, trainX)
# data_predict = read_data_fea(fea_list_CB, predictX)
# zhunquelvKNN=train_knn(data_sample, trainy, data_predict, predicty)
# #zhunquelvSVM=train_svm(data_sample, trainy, data_predict, predicty)
# #zhunquelvTREE=train_tree(data_sample, trainy, data_predict, predicty)
# weidu=1 - (len(fea_list_CB) / len(feature))
# print('准确率KNN:',zhunquelvKNN)
# #print('准确率SVM:',zhunquelvSVM)
# #print('准确率J48:',zhunquelvTREE)
# print('维度缩减：',weidu)
