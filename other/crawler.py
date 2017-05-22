# coding=utf-8
# 爬虫示例
from mpl_finance import _quotes_historical_yahoo
from datetime import date
from datetime import datetime
import pandas as pd
today = date.today()
start = (today.year - 1,today.month,today.day)
quotes = _quotes_historical_yahoo('AXP',start,today)
fields = ['date','open','close','high','low','volume']
list1 = []
for i in range(0,len(quotes)):
    x = date.fromordinal(int(quotes[i][0]))
    y = datetime.strftime(x,'%Y-%m-%d')
    list1.append(y)
df = pd.DataFrame(quotes,index = list1,columns = fields)
df = df.drop(['date'],axis=1)
print df