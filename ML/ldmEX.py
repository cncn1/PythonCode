#!/usr/bin/python
# -*- coding:utf-8 -*-
import timeit
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from ML import util


# kTup: lin: 线性
# rbf: 高斯核 公式：exp{(xj - xi)^2 / (2 * δ^2)} | j = 1,...,N
# δ：有用户自设给出kTup[1]
def kernelTrans(X, A, kTup):  # calc the kernel or transform data to a higher dimensional space
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'line':
        K = X * A.T  # linear kernel
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))  # divide in NumPy is element-wise not matrix like Matlab
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


# sampleX 抽样的数据
# sampleX_test 未被抽样的数据 (需注意应有sampleX << sampleX_test,这是算法的重点，否则算法没多大意义了)
# kTup核函数的类型与所需参数列表
def nystrom(sampleX, sampleX_test, kTup):
    sampleX_Num = np.shape(sampleX)[0]
    sampleX_testNum = np.shape(sampleX_test)[0]
    A = np.mat(np.zeros((sampleX_Num, sampleX_Num)))
    B = np.mat(np.zeros((sampleX_Num, sampleX_testNum)))
    sampleX = np.mat(sampleX)
    sampleX_test = np.mat(sampleX_test)
    for i in range(sampleX_Num):  # 获得核函数A
        A[:, i] = kernelTrans(sampleX, sampleX[i, :], kTup)
    for i in range(sampleX_testNum):  # 获得核函数B
        B[:, i] = kernelTrans(sampleX, sampleX_test[i, :], kTup)
    C = B.T * A.I * B  # nystrom方法估计C
    W1 = np.hstack((A, B))
    W2 = np.hstack((B.T, C))
    W = np.vstack((W1, W2))  # 最终拼接完的核函数矩阵
    return W


class optStruct:
    # @dataMatIn: 数据集,type: mat
    # @classLabels: 类标签,type: mat
    # @C：自设调节参数
    # @toler: 自设容错大小
    # @kTup: 核函数类型
    def __init__(self, dataMatIn, classLabels, C, toler, kTup, **kw):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))  # 初始化一个m的列向量α
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))  # 误差(Ei)缓存 #first column is valid flag
        self.K = np.mat(np.zeros((self.m, self.m)))  # 初始化一个存储核函数值得m*m维的K
        if kw == {}:
            for i in range(self.m):  # 获得核函数的值K(X,xi)
                self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)
        else:
            self.K = nystrom(kw['sampleX'], kw['sampleX_test'], kTup)


def ldm(dataMatIn, classLabels, lambda1, lambda2, C=200, toler=0.01, maxIter=10000, kTup=('line', 0), **kw):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup, **kw)
    m, n = np.shape(dataMatIn)
    beta = np.mat(np.zeros((m, 1)))
    betaOld = np.mat(np.zeros((m, 1)))
    Y = np.mat(np.zeros((m, m)))
    for i in xrange(m):
        Y[i, i] = classLabels[i, 0]
    Gy = oS.K * classLabels
    GY = oS.K * Y
    YG = Y * oS.K
    Q = 4 * lambda1 * (m * oS.K.T * oS.K - Gy * Gy.T) / (m ** 2) + oS.K
    QI = Q.I
    oS.alphas = lambda2 * QI * Gy / m
    A = QI * GY
    e = np.mat(np.eye(m))
    count = 0
    flag = False
    while count < maxIter:
        if flag:
            break
        for i in xrange(m):
            hii = e[i] * YG * QI * GY * e[i].T
            dfbetai = e[i] * YG * oS.alphas - 1
            if abs(dfbetai) < toler:
                flag = True
                break
            betaOld[i, 0] = beta[i, 0]
            if hii != 0:
                beta[i, 0] = min(max(beta[i, 0] - 1.0 * dfbetai / hii, 0), C)
            else:
                beta[i, 0] = C
            oS.alphas = oS.alphas + (beta[i, 0] - betaOld[i, 0]) * A * e[i].T
        count += 1
    if kw == {}:
        print 'ldm迭代%d次收敛' % count
    else:
        print 'nyldm迭代%d次收敛' % count
    return oS.alphas, oS.K


# ldm原始升级版
# dataArr 数据列表
# labelArr 数据类别列表
# lambda1 数据均值参数
# lambda2 数据方差参数
# C 软间隔容错参数
# toler 迭代收敛范围
# maxIter 最大迭代次数
# kTup 核函数的类型与所需参数列表
# kw 可变参数,主要用来接收是否应用nystrom方法的抽样参数
def ldmEXtrain(dataArr, labelArr, lambda1, lambda2, C=200, toler=0.01, maxIter=10000, kTup=('line', 0), **kw):
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr)  # 这里如果不用np.loadtxt的方法读取数据需要写成 labelMat = mat(labelArr).transpose()
    m, n = np.shape(datMat)
    alphas, K = ldm(datMat, labelMat, lambda1, lambda2, C, toler, maxIter, kTup, **kw)
    errorCount = 0
    # dataPredict = zeros((m,1))    # 记录训练数据分类的数组
    for i in range(m):
        predict = K[i] * alphas
        # dataPredict[i, 0] = sign(predict)
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    # print dataPredict           # 记录预测数据结果
    if kw == {}:
        print "the ldm_training error rate is: %f" % (float(errorCount) / m)
    else:
        print "the nyldm_training error rate is: %f" % (float(errorCount) / m)


def main(X, Y, low, high):
    while low < 499000:
        high = low + len - 1
        totStart = timeit.default_timer()
        time_stamp = datetime.datetime.now()
        print "time_stamp       " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S')  # strftime可以自定义时间的输出格式，
        print low, "-", high
        # x0, y0 = util.loadData('alpha', low=low, len=len)
        x0, y0 = X[low:high], Y[low:high]
        # print x0[0:5], y0[0:5]
        x0 = util.autoNorm(x0)  # 数据归一化

        # m, n = np.shape(x0)
        # index = np.array([i for i in range(m)]).reshape((m, 1))
        # x0 = np.hstack((index, x0))
        # y0 = np.hstack((index, y0))

        sampleX, sampleX_test, sampleY, sampleY_test = train_test_split(x0, y0, random_state=1, test_size=0.90)  # 数据抽样
        x = np.vstack((sampleX, sampleX_test))  # 将经过抽样处理的数据,按新的顺序组织数据
        y = np.vstack((sampleY, sampleY_test))
        start = timeit.default_timer()
        ldmEXtrain(x, y, 1000, 200000, 15, 0.0001, 30, kTup=('rbf', 1.5))
        end = timeit.default_timer()
        print 'ldm运行时间:', end - start
        start = timeit.default_timer()
        ldmEXtrain(x, y, 1000, 200000, 15, 0.0001, 30, kTup=('rbf', 1.5), sampleX=sampleX, sampleX_test=sampleX_test)
        end = timeit.default_timer()
        print 'nyldm运行时间:', end - start
        totEnd = timeit.default_timer()
        print '程序总运行时间:', totEnd - totStart
        low += len


if __name__ == '__main__':
    # low = 0;len = 500
    # X, Y = util.loadData('alpha')
    # main(X, Y, low, len)
    X, Y = util.loadData('news20')

