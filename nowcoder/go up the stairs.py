# coding=utf-8
# 上楼梯问题,斐波那契数列的应用


def fibonacci(n):
    x, y = 1, 1
    while n:
        x, y, n = y, x+y, n-1
    return x

try:
    while True:
        N = int(raw_input())
        print fibonacci(N)
except:
    pass
