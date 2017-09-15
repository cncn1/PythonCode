#!/usr/bin/python
# -*- coding:utf-8 -*-
from copy import deepcopy
from mpmath import inf
import sys

sys.stdin = open("input.txt")


def floyd_warshall(G):
    D = deepcopy(G)
    lengthD = len(D)  # 邻接矩阵大小
    P = [([i for i in xrange(lengthD)]) for j in range(lengthD)]
    for i in xrange(lengthD):
        for j in xrange(lengthD):
            if D[i][j] != inf:
                P[i][j] = i
    for i in range(lengthD):
        for j in range(lengthD):
            for k in range(lengthD):
                if D[i][j] > D[i][k] + D[k][j]:  # 两个顶点直接较小的间接路径替换较大的直接路径
                    P[i][j] = P[k][j]  # 记录新路径的前驱
                    D[i][j] = D[i][k] + D[k][j]
    return D, P  # D记录所有点间的最短路径，P记录最短路径终节点的前驱节点， 若转矩阵形式可用返回array(D) 和 array(P)


while True:
    try:
        p, q = map(int, raw_input().split())
        r = map(str, raw_input().split())
        index = 0
        dic = {}
        G = [([inf] * p) for i in range(p)]
        for i in range(p):
            G[i][i] = 0
        for i in range(0, len(r) - 1, 2):
            if not r[i] in dic:
                dic[r[i]] = index
                index += 1
            if not r[i + 1] in dic:
                dic[r[i + 1]] = index
                index += 1
            G[dic[r[i]]][dic[r[i + 1]]] = G[dic[r[i + 1]]][dic[r[i]]] = 1
        D, P = floyd_warshall(G)
        print G, '\n', D, '\n', P  # 信息输出
        dis = -1  # 表示隔了度
        for i in xrange(p):
            for j in xrange(p):
                dis = max(dis, D[i][j])
        if dis != inf:
            print dis, '\n'
        else:
            print 'DISCONNECTED\n'
    except:
        break
