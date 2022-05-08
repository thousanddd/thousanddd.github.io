---
layout: post
title:  "1.1. 时间序列建模基础"
date:   2022-04-18 
categories: 时间序列
---


### 前言


在股市做预测，从来都是很难的事情，个股价格波动受太多因素影响。但预测股市的波动，却是可行的。这里我们将用A股数据为例，结合机器学习和深度学习算法，做些尝试。


作为实验起步，我们仍从预测个股价格开始，这将通过时间序列建模的方式完成。之所以用时间序列作为基础进行数据分析和建模，是因为很多过往研究中，我们看到时间序列中的值存在相关性。可以通过对历史观测值和当前值的相关性建模，从而找到规律预测未来值。分析时间序列由4部分组成。包括：trend, seasonality, cyclicality 和 residual。下面我们以平安银行（000001.SZ）为例，具体演示下。先导入几个python模块。


```python
import akshare as ak
import numpy as np
import pandas as pd
import datetime
import os
import time
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
```


以“平安银行”（000001.SZ）为例，下载最近一年的历史日线数据。


```python
ticker='sz000001'
date1=datetime.date.today()+datetime.timedelta(-365)
date2=datetime.date.today()
startDate=date1.strftime('%Y%m%d')
endDate=date2.strftime('%Y%m%d')
sz000001Daily=ak.stock_zh_a_daily(symbol=ticker, start_date=startDate, end_date=endDate, adjust="qfq")
sz000001Close=sz000001Daily.close
```


简单看下日线的走势。


```python
x=sz000001Daily.date
y=sz000001Close
plt.figure(figsize=(10, 6))
plt.plot(x,y)
plt.title('000001.SZ Daily Close Prices')
plt.ylabel('Rmb')
plt.xlabel('Date')
plt.show()
```


![output_5_0.png](https://s2.loli.net/2022/04/18/FnSNatkLT75UpbC.png)

​    

但基础走势图无法做进一步的描述性分析，需要将数据分解成长期趋势项、季节性周期项以及残差项几部分。用下面现成的函数就可以做到。


```python
seasonal_decompose(sz000001Close, period=12).plot()
plt.show()
```


![output_7_0.png](https://s2.loli.net/2022/04/18/6OuwtK1QqjLZVId.png)


接着通过两个统计分析对时间序列进行分析，分别是：
* Autocroorelation Function 
* Partial Autocorrelation Function 


其中，Autocroorelation function（ACF）主要用来分析时间序列中的当前值和前序值之间的关系。因为在时间序列分析中，我们通常假设数据中的依赖结构不随时间变化。在这个假设下，影响协方差的唯一因素是两个时间序列中随机变量的距离 ，这个距离通常称为滞后lag，找到合适的lag很重要，是后续建模的基础。


```python
sm.graphics.tsa.plot_acf(sz000001Close, lags=30)
plt.xlabel('Number of Lags')
plt.show()
```


![output_9_0.png](https://s2.loli.net/2022/04/18/mRz2oASJ7VgNUf5.png)


从上面这个图可以看出，平安银行股价的现价$x_t$和前序价格$x_(t-h)$ 有显著的相关性。因为第一根垂直线代表了当前值和自己的相关系数，为1，到$lag=15$的时候，我们看到相关系数也在0.75以上，而且这样的相关性衰减的也比较平缓。从这个统计结果上也佐证了平安银行股价，在过去的一年内是具有显著趋势的，比如，这里的下跌走势。


另一个用于分析时间序列相关性的统计函数是Partial autocorrelation function （PACF)，偏自相关函数描述的是一组时间序列和它前面间隔n个时刻的一组时间序列之前的偏相关性。这里的偏相关性可以从本质上理解为去除了样本之间的干涉，也就是更早时刻的相关性影响。我们用实际的数据看下效果。


```python
sm.graphics.tsa.plot_pacf(sz000001Close, lags=30)
plt.xlabel('Number of Lags')
plt.show()
```


![output_12_0.png](https://s2.loli.net/2022/04/18/AdeIVhHxjFPctnp.png)


这里主要看下极端值的情况，就是超过蓝色区域（置信区间）的值，上面显示出的为$lag=6$和$lag=25$。


### 平稳性分析


接着看下平稳性分析。我们说一个时间序列是平稳的，是指该时间序列的一系列统计属性，比如，均值、方差、协方差等并不随着时间而发生变化。之所以关注时间序列是否平稳，主要还是基于建模的考量。我们在做预测的时候，往往是假定该时间序列的数据是遵循某种统计分布的，比如：我们常用的正态分布。但如果数据的分布随着时间变化，我们就很难去建模和预测未来的结果。


所以，当市场出现结构性风险，比如金融危机、地缘战争、或者“新冠”疫情的时候，平稳性分析就显得更加重要，值得我们把数据看的更加仔细。平稳性又可以进一步划分为：弱平稳、和强平稳。具体定义的数学公式就不写了，我们重点关注看下如何做平稳性的统计分析。这里用到的统计分析是：Dickey-Fuller (ADF)测试。在95%的置信度区间下，我们看到目标个股的收盘价数据序列，是非平稳的。


```python
statTest=adfuller(sz000001Close)[0:2]
print("The test statistic and p-value of ADF test are {}"
      .format(statTest))
```

    The test statistic and p-value of ADF test are (-1.281067068919137, 0.6377890306118901)


通过取时间序列中的当前值和其前一个值的差，即：$x_t$和$x_t-1$的差可以平划数据里的非平稳性。感觉从下面这个图，可以看到数据是围绕某个均线在上下波动了。然后，我们再重新跑一遍ADF测试，看看$p$ value是不是显著。答案是，显著的，对应一个极低的值（3.5876828130207457e-12）。


```python
diff_sz000001Close=sz000001Close.diff()
```


```python
plt.figure(figsize=(10, 6))
plt.plot(sz000001Daily.date,diff_sz000001Close)
plt.title('Differenced SZ00001 Close Price')
plt.ylabel('$')
plt.xlabel('Date')
plt.show()
```


![output_18_0.png](https://s2.loli.net/2022/04/18/V1PUTBKbDa79cW2.png)


```python
sm.graphics.tsa.plot_acf(diff_sz000001Close.dropna(),lags=30)
plt.xlabel('Number of Lags')
plt.show()
```


![output_19_0.png](https://s2.loli.net/2022/04/18/eUk82xTNW6dpmOi.png)


```python
statTest2=adfuller(diff_sz000001Close.dropna())[0:2]
print("The test statistic and p-value of ADF test after differencing are {}"\
      .format(statTest2))
```

    The test statistic and p-value of ADF test after differencing are (-7.626044383938417, 2.0674409027489904e-11)
