# coding=utf-8
# A+B

import sys

for line in sys.stdin:
    A, B = line.split(' ')
    A = ''.join(A.split(','))
    B = ''.join(B.split(','))
    print int(A) + int(B)