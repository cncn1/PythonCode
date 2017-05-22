# coding=utf-8
# 最小年龄的3个职工

import sys
sys.stdin = open("input.txt")
while True:
    try:
        x = []
        for i in xrange(input()):
            i = raw_input().split()
            x.append(i)
        y = sorted(x, key=lambda a: (int(a[2]), int(a[0]), a[1]))[:3]
        for j in y:
            print ' '.join(j)
    except EOFError:
        break