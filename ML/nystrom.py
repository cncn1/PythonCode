#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import similarity as sim
from sklearn.model_selection import train_test_split


def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


def GramMatrix(sampleA, sampleB, f=sim.euclidean):  # 相似度矩阵
    m, n = len(sampleA), len(sampleB)
    matrix = np.zeros((m, n))
    for i in xrange(m):
        for j in xrange(n):
            matrix[i][j] = f(sampleA[i], sampleB[j])
    return np.mat(matrix)

if __name__ == '__main__':
    path = '..//Datas/testSet.txt'  # 数据文件路径
    data = np.loadtxt(path, dtype=float, delimiter='\t')
    x0, y0 = np.split(data, (-1,), axis=1)
    # m, n = np.shape(x0)
    # index = np.array([i for i in range(m)]).reshape((m, 1))
    # x0 = np.hstack((index, x0))
    # y0 = np.hstack((index, y0))
    x_train, x_test, y_train, y_test = train_test_split(x0, y0, random_state=1, test_size=0.2)
    sampleX, sampleX_test, sampleY, sampleY_test = train_test_split(x0, y0, random_state=1, test_size=0.8)
    x = np.vstack((sampleX, sampleX_test))
    y = np.vstack((sampleY, sampleY_test))
    A = GramMatrix(sampleX, sampleX, sim.euclidean)
    B = GramMatrix(sampleX, sampleX_test, sim.euclidean)
    C = B.T * A.I * B
    W1 = np.hstack((A, B))
    W2 = np.hstack((B.T, C))
    W = np.vstack((W1, W2))
