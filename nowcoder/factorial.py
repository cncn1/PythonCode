# coding=utf-8
# 阶乘


def factorial(n):
    return reduce(lambda x, y: x*y, xrange(1, n+1))

try:
    while True:
        num = int(raw_input())
        oddSum = sum(factorial(x) for x in xrange(1, num+1, 2))
        evenSum = sum(factorial(x) for x in xrange(2, num+1, 2))
        print oddSum, evenSum
except:
    pass