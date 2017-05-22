# coding=utf-8
# 排序
import sys
sys.stdin = open("input.txt")

while True:
    try:
        n = int(raw_input())
        x = raw_input().split()
        y = sorted(x, key=lambda a: int(a))
        print ' '.join(y) + ' '
    except EOFError:
        break