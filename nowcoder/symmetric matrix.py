# coding=utf-8
# 对称矩阵
while True:
    try:
        a = []
        for _ in xrange(input()):
            a.append(tuple(raw_input().split()))
        print 'Yes!' if a == zip(*a) else 'No!'
    except:
        break