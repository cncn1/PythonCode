# coding=utf-8
# 最大的两个数
import sys
sys.stdin = open("input.txt")


def enhance_max(x):  # '查找列表中的最大值和次大值'
    largest = x[0]
    second_largest = x[1]
    if largest < second_largest:
        largest, second_largest = second_largest, largest
    for u in x[2:]:
        if u > largest:
            largest, second_largest = u, largest
        elif u > second_largest:
            second_largest = u
    return largest, second_largest

for i in xrange(input()):
    a = [map(int, raw_input().strip().split()) for j in xrange(4)]  # '生成二维数组'
    b = zip(*a)
    r = []
    for k in b:
        first, second = enhance_max(k)  # 'first表示查到的最大值,second表示找到的第二大值'
        firstIndex = k.index(first)  # 'firstIndex表示找到的最大值的位置'
        secondIndex = k.index(second)  # 'secondIndex'表示找到的第二大值的位置
        if firstIndex > secondIndex:
            firstIndex, secondIndex = secondIndex, firstIndex
        r.append([k[firstIndex], k[secondIndex]])
    for row in zip(*r):
        print ' '.join(map(str, row)) + ' '