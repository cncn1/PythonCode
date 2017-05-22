# coding=utf-8
# 矩阵最大值
while True:
    try:
        m, n = map(int, raw_input().split())
        for _ in xrange(m):
            r = map(int, raw_input().split())
            r[r.index(max(r))] = sum(r)
            print " ".join(map(str, r))
    except:
        break
