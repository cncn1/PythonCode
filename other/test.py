#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np

#
# y_hat = np.array([0.5, 1, 2, 0.4, 0.3, 0, 6])
# print y_hat
# c = y_hat > 0.5
# y_hat[c] = 1
# y_hat[~c] = 0
# print y_hat

# caonima = np.array([[-1, 0, 4, 1],
#                     [-5, 4, 0, -7],
#                     [3, 5, 0, 8],
#                     [-1, -4, 9, -3]])
# print np.linalg.det(caonima)

# import pandas
# iris = pandas.read_csv("iris.csv")
# king = 100
# Peter = king
# print iris.describe()

import pandas as pd
import bz2file
pathX = "F://Datas//Large Scale Data FTP//alpha//alpha_train.dat.bz2"
# infileX = bz2file.open(pathX, "r")
pathY = "F://Datas//Large Scale Data FTP//alpha//alpha_train.lab.bz2"
# infileY = bz2file.open(pathY, "r")
reader = pd.read_csv(pathX, iterator=True, header=None, sep=' ')
loop = True
chunkSize = 10000
chunks = []
count = 0
while loop:
    try:
        count += 1
        chunk = reader.get_chunk(chunkSize)
        chunks.append(chunk)
        print count
    except StopIteration:
        loop = False
        print "Iteration is stopped."
df = pd.concat(chunks, ignore_index=True)
# df = pd.DataFrame(columns='a')
print df.loc[0:5]
print df.loc[10:15]
print len(df)
# print df.loc[20000:20005]
