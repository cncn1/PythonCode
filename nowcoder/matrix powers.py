# coding=utf-8
# 矩阵幂
import numpy as np
import sys
sys.stdin = open("input.txt")


def matrixMul(A, B):
    res = [[0] * len(B[0]) for x in xrange(len(A))]
    for x in range(len(A)):
        for y in range(len(B[0])):
            for e in range(len(B)):
                res[x][y] += A[x][e] * B[e][y]
    return res


# 函数版
while True:
    try:
        for _ in xrange(input()):
            n, k = raw_input().split()
            n = int(n)
            k = int(k)
            data = []
            for i in xrange(n):
                data.append(map(int, raw_input().split()))
            z = data
            for i in xrange(k-1):
                z = matrixMul(z, data)
            for j in z:
                print ' '.join(map(str, j))
    except EOFError:
        break

# numpy 计算版
# while True:
#     try:
#         for _ in xrange(input()):
#             n, k = raw_input().split()
#             n = int(n)
#             k = int(k)
#             data = []
#             for i in xrange(n):
#                 data.append(map(int, raw_input().split()))
#             x = np.mat(data)
#             y = x ** k
#             z = y.tolist()
#             for j in z:
#                 print ' '.join(map(str, j))
#     except EOFError:
#         break
