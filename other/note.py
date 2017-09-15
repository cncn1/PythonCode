#!/usr/bin/python
# -*- coding:utf-8 -*-

# scipy = s
# numpy = np
# pandas = pd
# other = o

# s.1.计算排列组合
from scipy.special import comb, perm

print perm(3, 2), comb(3, 2)

# np.1 建立矩阵并求特征值和特征向量
import numpy as np

a = np.mat([[4, 6, 0],
            [-3, -5, 0],
            [-3, -6, 1]])

lamda0 = np.linalg.eigvals(a)  # 只求特征值
print lamda0, '\n'
lamda, vector = np.linalg.eig(a)  # 求特征值和特征向量
print lamda, '\n', vector

# np.2 计算矩阵的行列式的值,逆,伴随
A = np.array([[1, -2, 1], [0, 2, -1], [1, 1, -2]])

A_abs = np.linalg.det(A)  # 求A的行列式的值
print A_abs

B = np.linalg.inv(A)  # 求A的逆
print B

A_bansui = B * A_abs  # 求A的伴随
print A_bansui
# np.3 计算阶乘
print np.math.factorial(125)
# np.4 tile 函数

# tile(a,x):   x是控制a重复几次的，结果是一个一维数组
# tile(a,(x,y))：   结果是一个二维矩阵，其中行数为x，列数是一维数组a的长度和y的乘积
# tile(a,(x,y,z)):   结果是一个三维矩阵，其中矩阵的行数为x，矩阵的列数为y，而z表示矩阵每个单元格里a重复的次数。

# pd.1 读取cvs文件
import pandas

iris = pandas.read_csv("iris.csv")
print iris.describe()
# o.1 解方程
from sympy import *

x = Symbol('x')
fx = 32 * x ** 3 * exp(-1.0 / 8 * 0.1 ** 2 * x) - 0.0999999
x1, x2 = solve(fx, x)
print x1, x2

# o.2 获取排列组合的全部情况
from itertools import combinations, permutations

s1 = list(permutations([1, 2, 3], 2))
s2 = list(combinations([1, 2, 3], 2))
print s1, '\n', s2

# o.3.1 copy.copy 浅拷贝 只拷贝父对象，不会拷贝对象的内部的子对象。
# o.3.2 copy.deepcopy 深拷贝 拷贝对象及其子对象
import copy

a = [1, 2, 3, 4, ['a', 'b']]  # 原始对象

b = a  # 赋值，传对象的引用
c = copy.copy(a)  # 对象拷贝，浅拷贝
d = copy.deepcopy(a)  # 对象拷贝，深拷贝

a.append(5)  # 修改对象a
a[4].append('c')  # 修改对象a中的['a', 'b']数组对象

print 'a = ', a
print 'b = ', b
print 'c = ', c
print 'd = ', d
