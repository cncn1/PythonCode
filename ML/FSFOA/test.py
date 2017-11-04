#!/usr/bin/python
# -*- coding:utf-8 -*-
import random
import sys
import numpy as np
import time

def index_replace(index, replace_string, const_value):
    new_string = ''
    for i in range(len(replace_string)):
        if i != index:
            new_string += replace_string[i]
        else:
            new_string += str(const_value)
    return new_string


# 连串反转方法
def lianchuan():
    a = '1' * 9
    index = 6
    print a
    print a[:index]
    print a[index + 1:]
    t = (int(a[index]) + 1) % 2
    a = a[:index] + str(t) + a[index + 1:]
    print a


# 数组反转方法
def revers(index):
    s = [1] * 9
    index = index
    s[index] = (s[index] + 1) % 2
    print s


# 删除方法
def delete_together(delete_index, a):
    for i in delete_index:
        a[i] = 'k'
    for i in xrange(len(delete_index)):
        a.remove('k')


class Tree:
    def __init__(self, tree_list, tree_age):
        self.list = tree_list
        self.age = tree_age


if __name__ == '__main__':
    start = time.clock()
    print start
    # k = ''.join(t)
    # print k
    # index = [1, 2]
    # dataMap = np.array([[1, 4, 1, 4],
    #                     [2, 5, 2, 5],
    #                     [3, 6, 3, 6]])
    # dataSim = []
    # print dataMap[:, 2]
    # for i in index:
    #     dataSim.append(dataMap[:, i])
    # print np.mat(dataSim).T