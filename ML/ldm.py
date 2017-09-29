#!/usr/bin/python
# -*- coding:utf-8 -*-
import timeit
from numpy import *
from ML import util


# @kTup: lin: 线性
# rbf: 高斯核 公式：exp{(xj - xi)^2 / (2 * δ^2)} | j = 1,...,N
# δ：有用户自设给出kTup[1]
def kernelTrans(X, A, kTup):  # calc the kernel or transform data to a higher dimensional space
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'line':
        K = X * A.T  # linear kernel
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))  # divide in NumPy is element-wise not matrix like Matlab
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


class optStruct:
    # @dataMatIn: 数据集,type: mat
    # @classLabels: 类标签,type: mat
    # @C：自设调节参数
    # @toler: 自设容错大小
    # @kTup: 核函数类型
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))  # 初始化一个m的列向量α
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))  # 误差(Ei)缓存 #first column is valid flag
        self.K = mat(zeros((self.m, self.m)))  # 初始化一个存储核函数值得m*m维的K
        for i in range(self.m):  # 获得核函数的值K(X,xi)
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def ldm(dataMatIn, classLabels, lambda1, lambda2, C=200, toler=0.01, maxIter=10000, kTup=('line', 0)):  # full Platt SMO
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    m, n = shape(dataMatIn)
    beta = mat(zeros((m, 1)))
    betaOld = mat(zeros((m, 1)))
    Y = mat(zeros((m, m)))
    for i in xrange(m):
        Y[i, i] = classLabels[i, 0]
    Gy = oS.K * classLabels
    GY = oS.K * Y
    YG = Y * oS.K
    Q = 4 * lambda1 * (m * oS.K.T * oS.K - Gy * Gy.T) / (m ** 2) + oS.K
    QI = Q.I
    oS.alphas = lambda2 * QI * Gy / m
    A = QI * GY
    e = mat(eye(m))
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
    print 'ldm迭代%d次收敛\n' % count
    return oS.alphas, oS.K

# 原始版ldm算法实现
# dataArr 数据列表
# labelArr 数据类别列表
# lambda1 数据均值参数
# lambda2 数据方差参数
# C 软间隔容错参数
# toler 迭代收敛范围
# maxIter 最大迭代次数
# kTup 核函数的类型与所需参数列表
# kw 可变参数,主要用来接收是否应用nystrom方法的抽样参数
def ldmtrain(dataArr, labelArr, lambda1, lambda2, C=200, toler=0.01, maxIter=10000, kTup=('line', 0)):
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    alphas, K = ldm(datMat, labelMat, lambda1, lambda2, C, toler, maxIter, kTup)
    errorCount = 0
    # dataPredict = zeros((m,1))    # 记录训练数据分类的数组
    for i in range(m):
        predict = K[i] * alphas
        # dataPredict[i, 0] = sign(predict)
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    # print dataPredict           # 记录预测数据结果
    print "the ldm_training error rate is: %f" % (float(errorCount) / m)


if __name__ == '__main__':
    dataArr, labelArr = util.loadDataSet('../Datas/testSet.txt', '\t')
    dataArr = util.autoNorm(array(dataArr))
    print dataArr
    start = timeit.default_timer()
    ldmtrain(dataArr, labelArr, 2000, 3600, 15, 0.0001, 200, kTup=('rbf', 1.5))
    end = timeit.default_timer()
    print 'ldm运行时间:', end - start
