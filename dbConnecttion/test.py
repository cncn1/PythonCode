#!/usr/bin/python
# -*- coding:utf-8 -*-
from MySqlConn import Mysql
import SqlUtil

# 申请资源
mysql = Mysql()

sqlAll = "SELECT sepallength as spl,sepalwidth as spw FROM iris"
result = mysql.getAll(sqlAll)
if result:
    print "get all"
    for row in result:
        print "%s\t%s" % (row["spl"], row["spw"])

result = mysql.getOne(sqlAll)
print "get one"
print "%s\t%s" % (result["spl"], result["spw"])

key_list = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']
sql = SqlUtil.get_s_sql('iris', key_list, {'class': 1})
r = mysql.getAll(sql)
print r

# 释放资源
mysql.dispose()
