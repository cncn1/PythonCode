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
print lamda0,'\n'
lamda, vector = np.linalg.eig(a)  # 求特征值和特征向量
print lamda, '\n', vector

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
