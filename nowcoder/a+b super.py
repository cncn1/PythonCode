# coding=utf-8
# 大整数a+b

import sys
sys.stdin = open("input.txt")
try:
    while True:
        a, b = raw_input().split()
        a = int(a)
        b = int(b)
        print a+b
except:
    pass