# coding=utf-8
# 矩阵乘法
import sys
sys.stdin = open("input.txt")


def matrixMul(A, B):
    res = [[0] * len(B[0]) for x in xrange(len(A))]
    for x in range(len(A)):
        for y in range(len(B[0])):
            for k in range(len(B)):
                res[x][y] += A[x][k] * B[k][y]
    return res


def matrixMul2(A, B):
    return [[sum(r * s for r, s in zip(r, s)) for s in zip(*B)] for r in A]

while True:
    try:
        a = []
        b = []
        for i in xrange(2):
            a.append(map(int, raw_input().strip().split()))
        for j in xrange(3):
            b.append(map(int, raw_input().strip().split()))
        c = matrixMul2(a, b)
        for d in c:
            print ' '.join(map(str, d)) + ' '
    except:
        break
