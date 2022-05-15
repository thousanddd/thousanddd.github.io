---
layout: post
title:  "2.5 基于移动平均策略下的回测实验：立讯精密（002475.SZ）"
date:   2022-05-14
categories: 移动平均 策略回测 VIX
---


### 策略回测


任何AI技术为基础的预测模型要跑赢市场并不是一件容易的事情，之前我们做过类似<u>“猴子都能选股”</u>的测试。结果往往可以让人大跌眼镜。所以对策略的回测是必要的环节，下面就是一些关于回测的实验。


和之前基于统计参数的建模顺序类似，我们先基于移动平均搭建策略并进行回测。这里选择的个股是“立讯精密”（002475.SZ）。我们先看下个股和大盘的走势关系。


```python
import akshare as ak
import numpy as np
import pandas as pd
from pylab import plt, mpl
from sklearn.metrics import accuracy_score
```


```python
ticker = 'sz002475'
sz002475_daily = ak.stock_zh_a_daily(symbol=ticker, start_date="20190101", end_date="20250101", adjust="qfq")
sz002475_close = sz002475_daily.close
sz002475_date = sz002475_daily.date
sz002475_close.shape
```

    (814,)


```python
ticker = 'sh000001'
sh000001_daily = ak.stock_zh_index_daily(symbol=ticker)
sh000001_close = sh000001_daily.close
sh000001_date = sh000001_daily.date
sh000001_close.shape
```

    (7670,)


```python
sz002475_nor = (sz002475_close - sz002475_close.mean())/sz002475_close.std()
sz002475_nor.shape
```

    (814,)


```python
start = len(sh000001_close)-len(sz002475_close)
end = len(sh000001_close)
sh000001_new = sh000001_close[start:end]
sh000001_nor = (sh000001_new - sh000001_new.mean())/sh000001_new.std()
sh000001_nor.shape
```

    (814,)


将收盘价标准化处理后，方便在同一坐标轴上展示，后面建模的时候数据也更好用。


```python
fig, ax=plt.subplots(figsize=(10,6))
ax.plot(sz002475_date, sz002475_nor, label='Normalized sz002475 Daily Price', linestyle='--')
ax.plot(sz002475_date, sh000001_nor, linestyle='solid', label="Normalized SSE Index Daily Price")
ax.set_title('Daily Close Price of SZ002475 and SSE Index')
ax.set(xlabel='Date', ylabel='Normalized CLose Price')
ax.legend(loc='best')
plt.grid()
```

    
![output_7_0.png](https://s2.loli.net/2022/05/15/5P9i7p1FVbr3sSk.png)
    

### 移动平均策略


移动平均策略的基本逻辑，是要通过两个MA线的交叉变化，决定多空的仓位。比如，我们这里设置的是，当短期的移动平均线在长期移动平均线的上面，我们买入这个股；反之我们就卖出这个股票，保持空仓（假设：我们不考虑融券卖空）。


我们用之前Autocroorelation function（ACF）的函数找到合适的滞后值。通过下面的图，我们分别设定为short-term=7，long-term=23。


```python
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
```


```python
sz002475_diff=sz002475_close.diff().dropna()
sm.graphics.tsa.plot_acf(sz002475_diff,lags=30)
plt.xlabel('ACF - SZ002475')
plt.show()
```

    
![output_10_0.png](https://s2.loli.net/2022/05/15/GYKHPyX5bUm7Wew.png)
    

```python
data = pd.DataFrame(sz002475_close, columns = ['close'])
data['sma1'] = data['close'].rolling(7).mean()
data['sma2'] = data['close'].rolling(23).mean()
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 814 entries, 0 to 813
    Data columns (total 3 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   close   814 non-null    float64
     1   sma1    808 non-null    float64
     2   sma2    792 non-null    float64
    dtypes: float64(3)
    memory usage: 19.2 KB
    

```python
data.plot(figsize=(10, 6))
plt.title('Daily Close Price of SZ002475 with Moving Average')
plt.ylabel('Price')
plt.xlabel('Days')
plt.grid()
```


![output_12_0.png](https://s2.loli.net/2022/05/15/Sm1BsU9T7rdDlEc.png)
    

上面红色线条=1的时候，标记我们开仓（买入）的时间和持有时长。我们假设了无法随时都能融到券卖空，所以空仓对应红色线条=0。


```python
data.dropna(inplace=True)
data['enter'] = np.where(data['sma1'] > data['sma2'], 1, 0)
data['enter'] = data['enter'].shift(1)
data.dropna(inplace=True)
data.plot(figsize=(10, 6), secondary_y='enter')
plt.title('Daily Close Price of SZ002475 With Enter and Exit')
plt.ylabel('Price')
plt.xlabel('Days')
plt.grid()
```

    
![output_14_0.png](https://s2.loli.net/2022/05/15/PphiA8XoU4FYkJz.png)
    

### 回测效果


我们用log return分别计算了从2019年1月1日，直接持有该股票，即：return对应的回报，和按照我们的移动平均买卖策略持有该股票，获得的收益，即：perform的值。


```python
data['return'] = np.log(data['close'] / data['close'].shift(1))
data.dropna(inplace=True)
data['perform'] = data['enter'] * data['return']
data[['return', 'perform']].sum().apply(np.exp)
```

    return     2.940661
    perform    4.120042
    dtype: float64


```python
data[['return', 'perform']].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.title('A MA Trading Strategy for SZ002475')
plt.ylabel('Net Value')
plt.xlabel('Days')
plt.grid()
```


![output_17_0.png](https://s2.loli.net/2022/05/15/5GexvWRBr1gkV26.png)
    

从上面这个业绩图我们可以看出：


1. 通过简单移动平均线的交叉判定买卖点的方法，并不能真正产生超额收益。这个策略唯一有用的地方，就是当个股下跌的时候，保持空仓，避免净值损失。
2. 当然，上述例子主要验证策略回测的基本逻辑，并没有涉及对波动率、价格的预测。
