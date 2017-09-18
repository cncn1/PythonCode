# coding=utf-8
# 题目：从1,2,3，......，98,99,2015这100个数中，任意选择若干个数（可能为0个数）求异或，试求异或的期望值
import random


def Sample(t, len):
    f = []
    for i in xrange(len):
        f.append(0)
    n = random.randint(0, 100)  # 生成[0,100]内的随机数，表示抽样个数
    while n > 0:
        m = random.randint(0, 99)  # 抽取那个位置上的数
        if f[m] == 0:
            f[m] = 1
            n -= 1
    s = 0
    for i in xrange(len):
        if f[i] == 1:
            s ^= f[i]
    return s

def main():
    a = []
    for i in xrange(99):
        a.append(i + 1)
    a.append(2015)
    total = 1000000  # 采样次数
    E = 0.0
    for i in xrange(total):
        E += Sample(a, a.__len__())
    print E

if __name__ == "__main__":
    main()
