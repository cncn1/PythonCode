# coding=utf-8
# 三角形的边

import sys

sys.stdin = open("input.txt")
while True:
    a, b, c = map(int, raw_input().split())
    if a == 0:
        break
    print a + b + c - 2 * max(a, b, c)
