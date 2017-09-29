#!/usr/bin/python
# -*- coding:utf-8 -*-
import timeit
from numpy import *
from sklearn.model_selection import train_test_split


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append(map(float, lineArr[:-1]))
        labelMat.append(float(lineArr[-1]))
    return dataMat, labelMat

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
            beta[i, 0] = min(max(beta[i, 0] - 1.0 * dfbetai / hii, 0), C)
            oS.alphas = oS.alphas + (beta[i, 0] - betaOld[i, 0]) * A * e[i].T
        count += 1
    print 'ldm迭代%d次收敛\n' % count
    return oS.alphas, oS.K


def ldmtrain(dataArr, labelArr, lambda1, lambda2, C=200, toler=0.01, maxIter=10000, kTup=('line', 0)):
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    alphas, K = ldm(datMat, labelMat, lambda1, lambda2, C, toler, maxIter, kTup)  # C=200 important
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
    # path = '.\\testSetRBF2.txt'  # 数据文件路径
    # data = loadtxt(path, dtype=float, delimiter='\t')
    # x, y = split(data, (2,), axis=1)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)
    dataArr, labelArr = loadDataSet('../Datas/testSetRBF.txt')
    start = timeit.default_timer()
    ldmtrain(dataArr, labelArr, 1000, 20000, 15, 0.01, 100, kTup=('rbf', 1.5))
    end = timeit.default_timer()
    print 'ldm运行时间:', end - start
