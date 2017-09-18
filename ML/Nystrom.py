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
    path = 'iris.data'  # 数据文件路径
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    x, y = np.split(data, (4,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)
    sampleX, sampleX_test, sampleY, sampleY_test = train_test_split(x, y, random_state=1, test_size=0.8)
    A = GramMatrix(sampleX, sampleX, sim.euclidean)
    B = GramMatrix(sampleX, sampleX_test, sim.euclidean)
    C = B.T * A.I * B
    print C
    # print sim.pearson(sampleX[2], sampleX[1])
