#!/usr/bin/python
# -*- coding:utf-8 -*-


# 计算欧几里德距离：
def euclidean(p, q):
    m, n = len(p), len(q)
    assert m == n
    # 计算欧几里德距离,并将其标准化
    e = sum([(p[i] - q[i]) ** 2 for i in xrange(n)])
    # 在日常使用中，一般习惯于将相似度与1类比，相似度在数值上反映为0 <= Similarity(X, y) <= 1，越接近1，相似度越高；
    # 那么我们在使用欧几里得距离时，可以通过1 /（1 + Distance(X, Y)）来贯彻上一理念。
    return 1 / (1 + e ** .5)


# 计算皮尔逊相关度：几个数据集中出现异常值的时候，欧几里德距离就不如皮尔逊相关度‘稳定’，它会在出现偏差时倾向于给出更好的结果。
def pearson(p, q):
    m, n = len(p), len(q)
    assert m == n
    # 分别求p，q的和
    sumx = sum([p[i] for i in xrange(n)])
    sumy = sum([q[i] for i in xrange(n)])
    # 分别求出p，q的平方和
    sumxsq = sum([p[i] ** 2 for i in xrange(n)])
    sumysq = sum([q[i] ** 2 for i in xrange(n)])
    # 求出p，q的乘积和
    sumxy = sum([p[i] * q[i] for i in xrange(n)])
    # print sumxy
    # 求出pearson相关系数
    up = sumxy - sumx * sumy / n
    down = ((sumxsq - sumxsq ** 2 / n) * (sumysq - sumysq ** 2 / n)) ** .5
    # 若down为零则不能计算，return 0
    if down == 0:
        return 0
    r = up / down
    return r


# 计算曼哈顿距离：
def manhattan(p, q):
    m, n = len(p), len(q)
    assert m == n
    # 计算曼哈顿距离
    distance = sum(abs(p[i] - q[i]) for i in xrange(n))
    return distance


# 计算jaccard系数 注意：在使用之前必须对两个数据集进行去重
def jaccard(p, q):
    # 求交集的两种方式
    # c = [i for i in p if i in q]
    c = list(set(p).intersection(set(q)))

    # 求并集
    d = list(set(p).union(set(q)))

    # 求差集的两种方式，在p中但不在q中
    # e = [i for i in p if i not in q]
    # e = list(set(p).difference(set(q)))

    return float(len(c)) / len(d)


if __name__ == '__main__':
    p = ['shirt', 'shoes', 'pants', 'socks']
    q = ['shirt', 'shoes']
    print jaccard(p, q)
