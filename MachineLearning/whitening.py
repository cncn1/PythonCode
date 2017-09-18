# encoding = utf-8
# 数据白化(预处理阶段过程，实际效果可能没有理论来的好)
import numpy as np
import math


def whitening(x):
    m = len(x)
    n = len(x[0])
    # 计算x*x'
    xx = [[0.0] * n for tt in xrange(n)]
    for i in xrange(n):
        for j in xrange(i, n):
            s = 0.0
            for k in xrange(m):
                s += x[k][i] * x[k][j]
            xx[i][j] = s
            xx[j][i] = s
    # 计算x*x'的特征值和特征向量
    lamda, egs = np.linalg.eig(xx)
    lamda = [1 / math.sqrt(d) for d in lamda]
    # 计算白化矩阵U'D^（-0.5）*U
    t = [[0.0] * n for tt in xrange(n)]
    for i in xrange(n):
        for j in xrange(n):
            t[i][j] = lamda[j] * egs[i][j]
    whiten_matrix = [[0.0] * n for tt in xrange(n)]
    for i in xrange(n):
        for j in xrange(n):
            s = 0.0
            for k in xrange(n):
                s += t[i][k] * egs[j][k]
            whiten_matrix[i][j] = s
    # 白化x
    wx = [0.0] * n
    for i in xrange(n):
        for j in xrange(n):
            s = 0.0
            for k in xrange(n):
                s += whiten_matrix[i][k] * x[j][k]
            wx[i] = s
        x[j] = wx[:]

