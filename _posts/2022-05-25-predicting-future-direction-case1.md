---
layout: post
title:  "2.6 预测大盘或个股的走势-线性回归模型"
date:   2022-05-25
categories: 大盘走势 回归模型 VIX
---


之前实验可以看到，预测市场或个股的具体点位是很难的一件事，如果还需要将预测的时间缩短到T+1，那几乎是不可能实现的。但是预测市场或个股走势的建模时可以不断优化的，下面我们先从最基础的回归模型开始，不断提升模型的复杂度。


### 数据预处理


这次我们用另一个指数：000016.SH - 上证50指数做为基础数据，预测它的日线走势，即：在T预测下一天的方向。


```python
import akshare as ak
import numpy as np
import pandas as pd
from pylab import plt, mpl
```


```python
symbol = 'sh000016'
market_daily=ak.stock_zh_index_daily(symbol)
market_daily.info
```


    <bound method DataFrame.info of             date      open      high       low     close      volume
    0     2004-01-02   996.996  1021.568   993.892  1011.347     8064653
    1     2004-01-05  1008.279  1060.898  1008.279  1060.801  1446818000
    2     2004-01-06  1059.141  1086.694  1059.095  1075.656  1699133400
    3     2004-01-07  1075.562  1095.841  1070.980  1086.303  1372941900
    4     2004-01-08  1087.680  1108.291  1082.509  1102.662  1078042700
    ...          ...       ...       ...       ...       ...         ...
    4461  2022-05-18  2765.276  2765.276  2720.919  2745.528  2531908300
    4462  2022-05-19  2711.454  2748.303  2708.814  2744.545  2597853900
    4463  2022-05-20  2759.835  2807.979  2759.835  2807.942  3466762100
    4464  2022-05-23  2813.933  2813.933  2769.701  2782.548  2848875300
    4465  2022-05-24  2784.654  2786.542  2733.697  2734.192  3260985000
    
    [4466 rows x 6 columns]>


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
    4461  2745.528 -0.004075
    4462  2744.545 -0.000358
    4463  2807.942  0.022837
    4464  2782.548 -0.009085
    4465  2734.192 -0.017531
    
    [4465 rows x 2 columns]>


在测算基准值时，我们仍然用过去5天的历史数据，预测第6天的数，$lags = 5$。下面的数据转换很重要，因为需要将每5天历史数据拆成1条样本、存储在矩阵的一行，里面的每列对应一个输入变量$X_i$。再将第6天的数据单独存储为一列，即$Y$变量。


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
      <th>4461</th>
      <td>0.013268</td>
      <td>-0.010178</td>
      <td>0.009677</td>
      <td>-0.007125</td>
      <td>0.008375</td>
    </tr>
    <tr>
      <th>4462</th>
      <td>-0.004075</td>
      <td>0.013268</td>
      <td>-0.010178</td>
      <td>0.009677</td>
      <td>-0.007125</td>
    </tr>
    <tr>
      <th>4463</th>
      <td>-0.000358</td>
      <td>-0.004075</td>
      <td>0.013268</td>
      <td>-0.010178</td>
      <td>0.009677</td>
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
  </tbody>
</table>
<p>4460 rows × 5 columns</p>
</div>


```python
data['return']
```


    6       0.031854
    7      -0.013030
    8      -0.020734
    9      -0.005333
    10      0.008968
              ...   
    4461   -0.004075
    4462   -0.000358
    4463    0.022837
    4464   -0.009085
    4465   -0.017531
    Name: return, Length: 4460, dtype: float64


### 获得基准值


我们通过回归模型获得预测结果，将是我们的基准值。这里要指出的是，由于我们是预测第二天的走势，而不是具体涨跌幅，所以这里的输入变量是上面的5天的涨跌幅，而输出变量是第6天的变动方向，即$sign(return)$，而不是$return$值本身。 


我们用Numpy.linalg.lstsq(a, b, rcond=’warn’)函数返回线性矩阵方程，其中：

1. a：它描述了系数矩阵。
2. b：它描绘了标准或“因变量”值。 如果参数是二维矩阵，则为该特定矩阵的K列中的每列计算最小二乘。
3. Rcond：它是float数据类型。 对于较小的a的奇异值，基本上是一个截止比率。 在等级确定过程中，如果奇异值小于r乘以a的最大奇异值的乘积，则将其视为零。


```python
reg = np.linalg.lstsq(data[cols], np.sign(data['return']), rcond=None)[0]
```


```python
reg
```


    array([-1.28026913, -0.26106754,  0.99876739,  0.36559451, -0.32644208])


```python
data['prediction'] = np.sign(np.dot(data[cols], reg))
```


```python
data['prediction'].value_counts()
```


     1.0    2248
    -1.0    2212
    Name: prediction, dtype: int64


```python
hits = np.sign(data['return'] * data['prediction']).value_counts()
```

```python
hits
```


     1.0    2292
    -1.0    2168
    dtype: int64


```python
hits.values[0] / sum(hits)
```


    0.5139013452914798


上面的计算逻辑很直观，只有当1&1，或-1&-1的时候预测为正确的。这里是2292次，对应的预测准确率为51%。就是我们的基准值，后面的模型需要超过这个值才算成功。
