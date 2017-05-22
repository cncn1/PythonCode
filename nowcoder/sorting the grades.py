# coding=utf-8
# 成绩排序

import sys
sys.stdin = open("input.txt")
while True:
    try:
        x = []
        for _ in xrange(input()):
            x.append(raw_input().split())
        y = sorted(x, key=lambda a: (int(a[2]), a[0], a[1]))
        for j in y:
            print ' '.join(j)
    except EOFError:
        break