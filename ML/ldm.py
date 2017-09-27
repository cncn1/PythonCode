#!/usr/bin/python
# -*- coding:utf-8 -*-
import time
from numpy import *
from sklearn.model_selection import train_test_split


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    j = i  # we want to select any J not equal to i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


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


def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
            if k == i:
                continue  # don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:  # in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):  # after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
                (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  # this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            # print "L==H"
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # changed for kernel
        if eta >= 0:
            # print "eta>=0"
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # added this for the Ecache
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            # print "j not moving enough"
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])  # update i by the same amount as j
        updateEk(oS, i)  # added this for the Ecache the update is in the oppostie direction
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
            oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
            oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C=200, toler=0.0001, maxIter=10000, kTup=('line', 0)):  # full Platt SMO
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                # print "fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)
            iter += 1
        else:  # go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                # print "non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)
            iter += 1
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif alphaPairsChanged == 0:
            entireSet = True
            # print "iteration number: %d" % iter
    return oS.alphas, oS.b, oS.K


def calcWs(alphas, dataArr, classLabels):  # 计算w
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def train(dataArr, labelArr, C=200, toler=0.0001, maxIter=10000, kTup=('line', 0)):
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    alphas, b, K = smoP(dataArr, labelArr, C, toler, maxIter, kTup)  # C=200 important
    svInd = nonzero(alphas.A > 0)[0]  # 支持向量序号对应的列表
    svK = K[svInd]  # 全部支持向量对应的核矩阵
    labelSV = labelMat[svInd]
    errorCount = 0
    # dataPredict = zeros((m,1))    # 记录训练数据分类的数组
    for i in range(m):
        predict = svK[:, i].T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
            # dataPredict[i, 0] = sign(predict)
    # print dataPredict
    print "the training error rate is: %f" % (float(errorCount) / m)
    # print "there are %d Support Vectors" % len(svInd) # 支持向量的个数
    # sVs = datMat[svInd]  # get matrix of only support vectors
    w = calcWs(alphas, dataArr, labelArr)
    return w
    # for i in range(m):
    #     wp = datMat[i] * mat(w) + b   # 根据计算的w进行预测
    #     print wp


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
    ei = mat(ones((m, 1)))
    hii = ei.T * YG * QI * GY * ei
    count = 0
    flag = False
    while count < maxIter:
        if flag:
            break
        for i in xrange(m):
            dfbetai = ei.T * YG * oS.alphas - 1
            if dfbetai < toler:
                flag = True
                break
            betaOld[i, 0] = beta[i, 0]

            beta[i, 0] = beta[i, 0] - 1.0 * dfbetai / hii
            print 'beta:', beta[i, 0], '\n'
            oS.alphas = oS.alphas + (beta[i, 0] - betaOld[i, 0]) * A * ei
        count += 1
        print '迭代第%d次\n' % count
        # print oS.alphas
    # print '迭代%d次收敛\n' % count
    # print oS.alphas
    return oS.alphas, oS.K


def ldmtrain(dataArr, labelArr, lambda1, lambda2, C=200, toler=0.01, maxIter=10000, kTup=('line', 0)):
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    alphas, K = ldm(datMat, labelMat, lambda1, lambda2, C, toler, maxIter, kTup)  # C=200 important
    svInd = nonzero(alphas.A > 0)[0]  # 支持向量序号对应的列表
    svK = K[svInd]  # 全部支持向量对应的核矩阵
    labelSV = labelMat[svInd]
    errorCount = 0
    # dataPredict = zeros((m,1))    # 记录训练数据分类的数组
    for i in range(m):
        predict = svK[:, i].T * multiply(labelSV, alphas[svInd])
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
            # dataPredict[i, 0] = sign(predict)
    # print dataPredict
    print "the training error rate is: %f" % (float(errorCount) / m)
    # print "there are %d Support Vectors" % len(svInd) # 支持向量的个数
    # sVs = datMat[svInd]  # get matrix of only support vectors
    w = calcWs(alphas, dataArr, labelArr)
    return w
    # for i in range(m):
    #     wp = datMat[i] * mat(w) + b   # 根据计算的w进行预测
    #     print wp


if __name__ == '__main__':
    # path = '.\\testSetRBF2.txt'  # 数据文件路径
    # data = loadtxt(path, dtype=float, delimiter='\t')
    # x, y = split(data, (2,), axis=1)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    # w = train(dataArr, labelArr, 100, 0.0001, 10000, kTup=('rbf', 1.5))
    ldmtrain(dataArr, labelArr, 5, 2, 1, 0.0001, 10, kTup=('rbf', 1.5))
