# coding=utf-8
# 守型数

import sys
sys.stdin = open("input.txt")
while True:
    try:
        n = int(raw_input())
        nn = str(n ** 2)
        n = str(n)
        print 'Yes!' if (nn[len(nn) - len(n):] == n) else 'No!'
    except EOFError:
        break