# coding=utf-8
# 遍历链表

import sys
sys.stdin = open("input.txt")
while True:
    try:
        n = int(raw_input())
        r = map(int, raw_input().split())
        r.sort()
        print " ".join(map(str, r))
    except EOFError:
        break