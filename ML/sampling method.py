#!/usr/bin/python
# -*- coding:utf-8 -*-
import random
import numpy as np


# 单纯随机抽样（simple random sampling）
# 将调查总体全部观察单位进行编号，再用抽签法随机抽取（不放回）部分观察单位组成样本。C（k,n）种组合
# 优点：操作简单，均数、率及相应的标准误计算简单。
# 缺点：总体较大时，难以一一编号。
def simple_sampling(dataMat, num):
    try:
        samples = random.sample(dataMat, num)
        return samples
    except:
        print('sample larger than population')


# 有放回随机抽样（Repetition Random Sampling）
def repetition_sampling(dataMat, number):
    sample = []
    for i in range(number):
        sample.append(dataMat[random.randint(0, len(dataMat) - 1)])
    return sample


# 系统抽样（systematic sampling）
# 又称机械抽样、等距抽样，即先将总体的观察单位按某一顺序号分成n个部分，依次用相等间距，从每一部分个抽取一个观察单位组成样本。
# 优点：易于理解，简便易行
# 缺点：总体有周期或增减趋势时，易产生偏性。（抽样之前，先shuffle?）
def systematic_sampling(dataMat, num):
    k = int(len(dataMat) / num)
    samples = [random.sample(dataMat[i * k:(i + 1) * k], 1) for i in range(num)]
    return samples


# 整群抽样(cluster sampling)
# 总体分群，再随机抽取几个群组成样本，群内全部调查。
# 优点：便于组织、节省经费。
# 缺点：抽样误差大于单纯随机抽样。

# 分层抽样（stratified sampling）
# 先按对观察指标影响较大的某种特征，将总体分为若干个类别，再从每一层内随机抽取一定数量的观察单位，合起来组成样本。有按比例分配和最优分配两种方案。
# 优点：样本代表性好，抽样误差减少。
# 以上四种基本抽样方法都属单阶段抽样，实际应用中常根据实际情况将整个抽样过程分为若干阶段来进行，称为多阶段抽样。

# 各种抽样方法的抽样误差一般是：整群抽样≥单纯随机抽样≥系统抽样≥分层抽样

def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


if __name__ == '__main__':
    path = 'iris.data'  # 数据文件路径
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    x, y = np.split(data, (4,), axis=1)
    simpleX = simple_sampling(x, 30)
    print simpleX
