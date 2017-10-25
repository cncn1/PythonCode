#!/usr/bin/python
# -*- coding:utf-8 -*-
import random
import sys



def index_replace(index, replace_string, const_value):
    new_string = ''
    for i in range(len(replace_string)):
        if i != index:
            new_string += replace_string[i]
        else:
            new_string += str(const_value)
    return new_string


# 连串反转方法
def lianchuan():
    a = '1' * 9
    index = 6
    print a
    print a[:index]
    print a[index + 1:]
    t = (int(a[index]) + 1) % 2
    a = a[:index] + str(t) + a[index + 1:]
    print a


# 数组反转方法
def revers(index):
    s = [1] * 9
    index = index
    s[index] = (s[index] + 1) % 2
    print s


if __name__ == '__main__':
    for i in xrange(1000):
        s = random.randint(0,3)
        print s
    revers(6)
