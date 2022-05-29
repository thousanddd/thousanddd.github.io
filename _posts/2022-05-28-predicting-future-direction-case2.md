---
layout: post
title:  "2.7 预测大盘走势（2/3）-逻辑回归模型"
date:   2022-05-28
categories: 大盘走势 回归模型 VIX
---


从之前实验可以看到，用线性回归模型预测上证50第二天的走势，效果并不好。在实际交易中我们是有方法提升模型效果的，这里就不再赘述。沿着线性回归的逻辑，自然就能想到将预测指数第二天走势的问题，看作是一个标准的二分类问题。通过搭建逻辑回归模型、并用1或-1代表第二天的涨或跌。下面我们具体演示下。


### 数据预处理


```python
import akshare as ak
import numpy as np
import pandas as pd
from pylab import plt, mpl
```


```python
symbol = 'sh000016'
market_daily=ak.stock_zh_index_daily(symbol)
```


```python
data = pd.DataFrame({'close': market_daily.close})
```


```python
data['return'] = np.log(data['close']/data['close'].shift(1))
data.dropna(inplace = True)
data.info
```


    <bound method DataFrame.info of          close    return
    1     1060.801  0.047741
    2     1075.656  0.013906
    3     1086.303  0.009849
    4     1102.662  0.014947
    5     1081.785 -0.019115
    ...        ...       ...
    4464  2782.548 -0.009085
    4465  2734.192 -0.017531
    4466  2743.417  0.003368
    4467  2744.898  0.000540
    4468  2763.072  0.006599
    
    [4468 rows x 2 columns]>



```python
lags = 5
```


```python
cols = []
for lag in range(1, lags + 1):
    col = f'lag_{lag}'
    data[col] = data['return'].shift(lag)
    cols.append(col)
data.dropna(inplace=True)
```


```python
data[cols]
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lag_1</th>
      <th>lag_2</th>
      <th>lag_3</th>
      <th>lag_4</th>
      <th>lag_5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>-0.019115</td>
      <td>0.014947</td>
      <td>0.009849</td>
      <td>0.013906</td>
      <td>0.047741</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.031854</td>
      <td>-0.019115</td>
      <td>0.014947</td>
      <td>0.009849</td>
      <td>0.013906</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.013030</td>
      <td>0.031854</td>
      <td>-0.019115</td>
      <td>0.014947</td>
      <td>0.009849</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.020734</td>
      <td>-0.013030</td>
      <td>0.031854</td>
      <td>-0.019115</td>
      <td>0.014947</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.005333</td>
      <td>-0.020734</td>
      <td>-0.013030</td>
      <td>0.031854</td>
      <td>-0.019115</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4464</th>
      <td>0.022837</td>
      <td>-0.000358</td>
      <td>-0.004075</td>
      <td>0.013268</td>
      <td>-0.010178</td>
    </tr>
    <tr>
      <th>4465</th>
      <td>-0.009085</td>
      <td>0.022837</td>
      <td>-0.000358</td>
      <td>-0.004075</td>
      <td>0.013268</td>
    </tr>
    <tr>
      <th>4466</th>
      <td>-0.017531</td>
      <td>-0.009085</td>
      <td>0.022837</td>
      <td>-0.000358</td>
      <td>-0.004075</td>
    </tr>
    <tr>
      <th>4467</th>
      <td>0.003368</td>
      <td>-0.017531</td>
      <td>-0.009085</td>
      <td>0.022837</td>
      <td>-0.000358</td>
    </tr>
    <tr>
      <th>4468</th>
      <td>0.000540</td>
      <td>0.003368</td>
      <td>-0.017531</td>
      <td>-0.009085</td>
      <td>0.022837</td>
    </tr>
  </tbody>
</table>
<p>4463 rows × 5 columns</p>
</div>


### 训练模型


首先，我们将全量数据拆分成训练集和测试集。我们预留了最近一年的数据作为样本外的测试集，也可以比较直观的看到这个分类算法在过去一年的效果。当然，这里存在两个明显的短板：

1. 我们没有考虑交易成本
2. 即使模型预测的结果正确，在实际交易中也无法真正赚到当天完整的涨幅/跌幅带来的利润。


```python
split = int(len(data)*0.95)
data_train = data[:split]
data_test = data[split:]
```


接着我们用scikit-learn这个API来训练模型。这个API在我们之前的实验中也使用过，它在基于python的机器学习建模中应用很广泛。


```python
from sklearn import linear_model
```


```python
lm = linear_model.LogisticRegression(C=1e7, solver='lbfgs', multi_class='auto', max_iter=1000)
```


```python
lm.fit(data_train[cols], np.sign(data_train['return']))
```


    LogisticRegression(C=10000000.0, max_iter=1000)


训练集的准确率：


```python
data_train['prediction'] = lm.predict(data_train[cols])
```


```python
data_train['prediction'].value_counts()
```


     1.0    3172
    -1.0    1067
    Name: prediction, dtype: int64


```python
hits = np.sign(data_train['return'] * data_train['prediction']).value_counts()
```


```python
hits.values[0] / sum(hits)
```


    0.5123849964614295


测试集的准确率：


```python
data_test['prediction'] = lm.predict(data_test[cols])
```


```python
data_test['prediction'].value_counts()
```


     1.0    172
    -1.0     52
    Name: prediction, dtype: int64


```python
hits = np.sign(data_test['return'] * data_test['prediction']).value_counts()
```


```python
hits.values[0] / sum(hits)
```


    0.5089285714285714


### 回测策略效果


用模型预测的准确率，不能直观的看到这个策略的效果。所以，我们用预测的市场走势，乘以预测当天实际的涨跌幅，得到这个策略更直观的业绩效果。这里，我们仅显示样本外的结果。


```python
data_test['strategy1'] = data_test['prediction'] * data_test['return']
```


```python
data_test[['return', 'strategy1']].sum().apply(np.exp)
```


    return       0.795967
    strategy1    0.875927
    dtype: float64


```python
data_test[['return', 'strategy1']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.title('逻辑回归模型预测上证50走势 - 样本外')
plt.ylabel('Rmb')
plt.xlabel('Days')
plt.grid()
```

    
![output_30_0.png](https://s2.loli.net/2022/05/28/JB7tG59KndX2ezi.png)
    

上面的结果，看起来比之前的线性回归模型要进步了一点，但还不够在实盘中使用。我们接着将通过深度学习算法，验证能否进一步提升预测的准确率。
