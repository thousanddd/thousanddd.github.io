---
layout: post
title:  "3.1 浅谈机器学习的超参数调优"
date:   2022-06-19
categories: 大盘走势 超参数 模型调优
---


超参数调优是机器学习、深度学习模型训练过程中非常重要的一个环节。我们用下面的实验，演示超参数调优在预测股价走势上的应用。


### 数据获取


我们继续使用000001.SH - 上证指数做为基础数据，预测它的日线走势，即：在T预测T+1的方向。


```python
import akshare as ak
from pylab import plt, mpl
```


```python
import numpy as np
import pandas as pd
```


```python
start_date = '19910101'
```


```python
sse_daily=oneCNMarket('sh000001',start_date)
sse_daily.shape
```


    (7685, 5)


具体看下数据中变量的值、以及全量数据集中的样本数等。


```python
sse_daily.tail()
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
      <th>times</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7680</th>
      <td>20220613</td>
      <td>3256.275</td>
      <td>3272.991</td>
      <td>3229.309</td>
      <td>3255.551</td>
    </tr>
    <tr>
      <th>7681</th>
      <td>20220614</td>
      <td>3224.214</td>
      <td>3289.134</td>
      <td>3195.819</td>
      <td>3288.907</td>
    </tr>
    <tr>
      <th>7682</th>
      <td>20220615</td>
      <td>3289.104</td>
      <td>3358.545</td>
      <td>3288.851</td>
      <td>3305.407</td>
    </tr>
    <tr>
      <th>7683</th>
      <td>20220616</td>
      <td>3306.835</td>
      <td>3319.689</td>
      <td>3277.531</td>
      <td>3285.385</td>
    </tr>
    <tr>
      <th>7684</th>
      <td>20220617</td>
      <td>3265.512</td>
      <td>3323.280</td>
      <td>3262.894</td>
      <td>3316.786</td>
    </tr>
  </tbody>
</table>
</div>


```python
data = sse_daily
data.drop(labels=['times'], axis=1, inplace=True)
```


```python
data['return'] = np.log(data['close']/data['close'].shift(1))
data.dropna(inplace = True)
data.tail()
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
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>return</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7680</th>
      <td>3256.275</td>
      <td>3272.991</td>
      <td>3229.309</td>
      <td>3255.551</td>
      <td>-0.008955</td>
    </tr>
    <tr>
      <th>7681</th>
      <td>3224.214</td>
      <td>3289.134</td>
      <td>3195.819</td>
      <td>3288.907</td>
      <td>0.010194</td>
    </tr>
    <tr>
      <th>7682</th>
      <td>3289.104</td>
      <td>3358.545</td>
      <td>3288.851</td>
      <td>3305.407</td>
      <td>0.005004</td>
    </tr>
    <tr>
      <th>7683</th>
      <td>3306.835</td>
      <td>3319.689</td>
      <td>3277.531</td>
      <td>3285.385</td>
      <td>-0.006076</td>
    </tr>
    <tr>
      <th>7684</th>
      <td>3265.512</td>
      <td>3323.280</td>
      <td>3262.894</td>
      <td>3316.786</td>
      <td>0.009512</td>
    </tr>
  </tbody>
</table>
</div>


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
data.tail()
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
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>return</th>
      <th>lag_1</th>
      <th>lag_2</th>
      <th>lag_3</th>
      <th>lag_4</th>
      <th>lag_5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7680</th>
      <td>3256.275</td>
      <td>3272.991</td>
      <td>3229.309</td>
      <td>3255.551</td>
      <td>-0.008955</td>
      <td>0.014066</td>
      <td>-0.007640</td>
      <td>0.006772</td>
      <td>0.001665</td>
      <td>0.012723</td>
    </tr>
    <tr>
      <th>7681</th>
      <td>3224.214</td>
      <td>3289.134</td>
      <td>3195.819</td>
      <td>3288.907</td>
      <td>0.010194</td>
      <td>-0.008955</td>
      <td>0.014066</td>
      <td>-0.007640</td>
      <td>0.006772</td>
      <td>0.001665</td>
    </tr>
    <tr>
      <th>7682</th>
      <td>3289.104</td>
      <td>3358.545</td>
      <td>3288.851</td>
      <td>3305.407</td>
      <td>0.005004</td>
      <td>0.010194</td>
      <td>-0.008955</td>
      <td>0.014066</td>
      <td>-0.007640</td>
      <td>0.006772</td>
    </tr>
    <tr>
      <th>7683</th>
      <td>3306.835</td>
      <td>3319.689</td>
      <td>3277.531</td>
      <td>3285.385</td>
      <td>-0.006076</td>
      <td>0.005004</td>
      <td>0.010194</td>
      <td>-0.008955</td>
      <td>0.014066</td>
      <td>-0.007640</td>
    </tr>
    <tr>
      <th>7684</th>
      <td>3265.512</td>
      <td>3323.280</td>
      <td>3262.894</td>
      <td>3316.786</td>
      <td>0.009512</td>
      <td>-0.006076</td>
      <td>0.005004</td>
      <td>0.010194</td>
      <td>-0.008955</td>
      <td>0.014066</td>
    </tr>
  </tbody>
</table>
</div>


### 增加新的变量


除了上述lagging指标外，我们再增加几个技术分析常用的指标，作为后续建模的变量。


```python
def SMA(close, n):
    SMA = close.rolling(n).mean().shift(1)
    return SMA
```


```python
def EMA(close, n):
    EMA = close.ewm(span=n, min_periods=n).mean().shift(1)
    return EMA
```


```python
def MIN(close, n):
    MIN = close.rolling(n).min().shift(1)
    return MIN
```


```python
def MAX(close, n):
    MAX = close.rolling(n).max().shift(1)
    return MAX
```


```python
def MOM(d_return, n):
    MOM = d_return.rolling(n).mean().shift(1)
    return MOM
```


```python
data['sma5'] = SMA(data['close'], lags)
```


```python
data['ema5'] = EMA(data['close'], lags)
```


```python
data['min5'] = MIN(data['close'], lags)
```


```python
data['max5'] = MAX(data['close'], lags)
```


```python
data['mom5'] = MOM(data['return'], lags)
```


```python
data.dropna(inplace=True)
```


```python
cols.extend(['sma5', 'ema5', 'min5', 'max5', 'mom5'])
```


```python
data.drop(labels=['open', 'high', 'low', 'close'], axis=1, inplace=True)
```


```python
data.tail()
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
      <th>return</th>
      <th>lag_1</th>
      <th>lag_2</th>
      <th>lag_3</th>
      <th>lag_4</th>
      <th>lag_5</th>
      <th>sma5</th>
      <th>ema5</th>
      <th>min5</th>
      <th>max5</th>
      <th>mom5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7680</th>
      <td>-0.008955</td>
      <td>0.014066</td>
      <td>-0.007640</td>
      <td>0.006772</td>
      <td>0.001665</td>
      <td>0.012723</td>
      <td>3253.1434</td>
      <td>3249.100456</td>
      <td>3236.372</td>
      <td>3284.834</td>
      <td>0.005517</td>
    </tr>
    <tr>
      <th>7681</th>
      <td>0.010194</td>
      <td>-0.008955</td>
      <td>0.014066</td>
      <td>-0.007640</td>
      <td>0.006772</td>
      <td>0.001665</td>
      <td>3256.9792</td>
      <td>3251.250637</td>
      <td>3238.954</td>
      <td>3284.834</td>
      <td>0.001182</td>
    </tr>
    <tr>
      <th>7682</th>
      <td>0.005004</td>
      <td>0.010194</td>
      <td>-0.008955</td>
      <td>0.014066</td>
      <td>-0.007640</td>
      <td>0.006772</td>
      <td>3266.4078</td>
      <td>3263.802758</td>
      <td>3238.954</td>
      <td>3288.907</td>
      <td>0.002888</td>
    </tr>
    <tr>
      <th>7683</th>
      <td>-0.006076</td>
      <td>0.005004</td>
      <td>0.010194</td>
      <td>-0.008955</td>
      <td>0.014066</td>
      <td>-0.007640</td>
      <td>3274.7306</td>
      <td>3277.670839</td>
      <td>3238.954</td>
      <td>3305.407</td>
      <td>0.002534</td>
    </tr>
    <tr>
      <th>7684</th>
      <td>0.009512</td>
      <td>-0.006076</td>
      <td>0.005004</td>
      <td>0.010194</td>
      <td>-0.008955</td>
      <td>0.014066</td>
      <td>3284.0168</td>
      <td>3280.242226</td>
      <td>3255.551</td>
      <td>3305.407</td>
      <td>0.002847</td>
    </tr>
  </tbody>
</table>
</div>


### 描述统计分析


我们再通过一些描述性统计分析，对现有的数据集形成初步的理解。


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7674 entries, 11 to 7684
    Data columns (total 11 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   return  7674 non-null   float64
     1   lag_1   7674 non-null   float64
     2   lag_2   7674 non-null   float64
     3   lag_3   7674 non-null   float64
     4   lag_4   7674 non-null   float64
     5   lag_5   7674 non-null   float64
     6   sma5    7674 non-null   float64
     7   ema5    7674 non-null   float64
     8   min5    7674 non-null   float64
     9   max5    7674 non-null   float64
     10  mom5    7674 non-null   float64
    dtypes: float64(11)
    memory usage: 719.4 KB
    

```python
data.describe()
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
      <th>return</th>
      <th>lag_1</th>
      <th>lag_2</th>
      <th>lag_3</th>
      <th>lag_4</th>
      <th>lag_5</th>
      <th>sma5</th>
      <th>ema5</th>
      <th>min5</th>
      <th>max5</th>
      <th>mom5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7674.000000</td>
      <td>7674.000000</td>
      <td>7674.000000</td>
      <td>7674.000000</td>
      <td>7674.000000</td>
      <td>7674.000000</td>
      <td>7674.000000</td>
      <td>7674.000000</td>
      <td>7674.000000</td>
      <td>7674.000000</td>
      <td>7674.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.000418</td>
      <td>0.000416</td>
      <td>0.000417</td>
      <td>0.000416</td>
      <td>0.000416</td>
      <td>0.000418</td>
      <td>2060.900489</td>
      <td>2060.902776</td>
      <td>2025.400857</td>
      <td>2094.793033</td>
      <td>0.000417</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.021965</td>
      <td>0.021964</td>
      <td>0.021964</td>
      <td>0.021964</td>
      <td>0.021964</td>
      <td>0.021964</td>
      <td>1091.203834</td>
      <td>1090.862010</td>
      <td>1073.293363</td>
      <td>1108.068225</td>
      <td>0.010556</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.179051</td>
      <td>-0.179051</td>
      <td>-0.179051</td>
      <td>-0.179051</td>
      <td>-0.179051</td>
      <td>-0.179051</td>
      <td>106.820000</td>
      <td>107.053561</td>
      <td>105.770000</td>
      <td>107.820000</td>
      <td>-0.062770</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.007283</td>
      <td>-0.007283</td>
      <td>-0.007283</td>
      <td>-0.007283</td>
      <td>-0.007283</td>
      <td>-0.007282</td>
      <td>1198.664000</td>
      <td>1197.342701</td>
      <td>1178.640000</td>
      <td>1223.076000</td>
      <td>-0.004015</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000638</td>
      <td>0.000636</td>
      <td>0.000636</td>
      <td>0.000634</td>
      <td>0.000634</td>
      <td>0.000636</td>
      <td>2018.989600</td>
      <td>2020.833306</td>
      <td>1991.950500</td>
      <td>2042.676000</td>
      <td>0.000428</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.008376</td>
      <td>0.008373</td>
      <td>0.008373</td>
      <td>0.008373</td>
      <td>0.008371</td>
      <td>0.008371</td>
      <td>2951.520850</td>
      <td>2949.920169</td>
      <td>2906.334000</td>
      <td>2995.061750</td>
      <td>0.004585</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.719152</td>
      <td>0.719152</td>
      <td>0.719152</td>
      <td>0.719152</td>
      <td>0.719152</td>
      <td>0.719152</td>
      <td>5994.983600</td>
      <td>5976.112786</td>
      <td>5903.264000</td>
      <td>6092.057000</td>
      <td>0.182139</td>
    </tr>
  </tbody>
</table>
</div>


```python
import matplotlib.pyplot as plt
data.hist(bins=50, figsize=(20,15))
plt.show()
```

    
![output_33_0.png](https://s2.loli.net/2022/06/19/XlPU5vqakoVM2Tb.png)
    

```python
corr_matrix = data.corr()
```


```python
corr_matrix['return'].sort_values(ascending=False)
```

    return    1.000000
    mom5      0.070480
    lag_1     0.044211
    lag_3     0.040468
    lag_4     0.036561
    lag_2     0.030962
    lag_5     0.017164
    max5     -0.034042
    ema5     -0.034615
    sma5     -0.034875
    min5     -0.035348
    Name: return, dtype: float64


仅从线性相关系数看，上面所有$X$变量中并没有和$Y$特别相关的。当然，在实际交易使用的信号变量会有所不同，此处仅用于演示目的，不影响后续的建模和超参数调优。


```python
data['direction'] = np.where(data['return'] > 0, 1, 0)
```


```python
data_num = data
```


```python
data_num[cols].tail()
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
      <th>sma5</th>
      <th>ema5</th>
      <th>min5</th>
      <th>max5</th>
      <th>mom5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7680</th>
      <td>0.014066</td>
      <td>-0.007640</td>
      <td>0.006772</td>
      <td>0.001665</td>
      <td>0.012723</td>
      <td>3253.1434</td>
      <td>3249.100456</td>
      <td>3236.372</td>
      <td>3284.834</td>
      <td>0.005517</td>
    </tr>
    <tr>
      <th>7681</th>
      <td>-0.008955</td>
      <td>0.014066</td>
      <td>-0.007640</td>
      <td>0.006772</td>
      <td>0.001665</td>
      <td>3256.9792</td>
      <td>3251.250637</td>
      <td>3238.954</td>
      <td>3284.834</td>
      <td>0.001182</td>
    </tr>
    <tr>
      <th>7682</th>
      <td>0.010194</td>
      <td>-0.008955</td>
      <td>0.014066</td>
      <td>-0.007640</td>
      <td>0.006772</td>
      <td>3266.4078</td>
      <td>3263.802758</td>
      <td>3238.954</td>
      <td>3288.907</td>
      <td>0.002888</td>
    </tr>
    <tr>
      <th>7683</th>
      <td>0.005004</td>
      <td>0.010194</td>
      <td>-0.008955</td>
      <td>0.014066</td>
      <td>-0.007640</td>
      <td>3274.7306</td>
      <td>3277.670839</td>
      <td>3238.954</td>
      <td>3305.407</td>
      <td>0.002534</td>
    </tr>
    <tr>
      <th>7684</th>
      <td>-0.006076</td>
      <td>0.005004</td>
      <td>0.010194</td>
      <td>-0.008955</td>
      <td>0.014066</td>
      <td>3284.0168</td>
      <td>3280.242226</td>
      <td>3255.551</td>
      <td>3305.407</td>
      <td>0.002847</td>
    </tr>
  </tbody>
</table>
</div>


### 拆分训练和测试集


```python
split = int(len(data_num)*0.8)
data_train = data_num[:split]
data_test = data_num[split:]
```


```python
data_train.shape
```


    (6139, 12)


```python
data_test.shape
```


    (1535, 12)


```python
mu, std = data_train[cols].mean(), data_train[cols].std()
```


```python
data_train_tr = (data_train[cols] - mu) / std
```


```python
data_test_tr = (data_test[cols] - mu) / std
```


```python
from sklearn.linear_model import LogisticRegression
```


```python
model = LogisticRegression()
```


```python
model.fit(data_train_tr, data_train['direction'])
```

    LogisticRegression()


### 检查模型效果


```python
from sklearn.model_selection import cross_val_score
```


```python
cross_val_score(model, data_train_tr, data_train['direction'], cv=5, scoring="accuracy")
```


    array([0.60749186, 0.52850163, 0.52687296, 0.52442997, 0.52322738])


```python
from sklearn.model_selection import cross_val_predict
```


```python
y_train_pred = cross_val_predict(model, data_train_tr, data_train['direction'], cv=5)
```


```python
from sklearn.metrics import confusion_matrix
```


```python
confusion_matrix(data_train['direction'], y_train_pred)
```

    array([[ 624, 2295],
           [ 516, 2704]], dtype=int64)



除了confustion matrix，还可以增加其它对准确率的评价指标。这些指标是分类算法模型常用的评价指标，这里就不再追溯其定义和计算公式了。仅展示该等指标在我们数据上的效果：


```python
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
```


```python
accuracy_score(data_train['direction'], y_train_pred)
```


    0.542107835152305


```python
recall_score(data_train['direction'], y_train_pred)
```

    0.8397515527950311


```python
precision_score(data_train['direction'], y_train_pred)
```

    0.5409081816363273


```python
f1_score(data_train['direction'], y_train_pred)
```

    0.6579875897311108


### 超参数调优


之所以讨论超参数，本质上是通过不同的超参数组合，找到最优的模型系数。需要说明的是，也只有这些参数会被用于预测，超参数本身被不会被模型使用。比如，我们在预测股价走势上，我们对预测结果是0或1我们并没有倾向性，所以此处将超参数调优的目标scoring设定为准确率。我们使用GridSearchCV的函数获得不同组合下的模型结果。


```python
from sklearn.model_selection import GridSearchCV
```


```python
grid_values = {'penalty': ['none', 'l1', 'l2'],'C':[0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 25]}
```


```python
grid_search = GridSearchCV(model, param_grid = grid_values, scoring = 'accuracy')
```


```python
grid_search.fit(data_train_tr, data_train['direction'])
```


    GridSearchCV(estimator=LogisticRegression(),
                 param_grid={'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 25],
                             'penalty': ['none', 'l1', 'l2']},
                 scoring='accuracy')


```python
y_pred_search = grid_search.predict(data_train_tr)
```


```python
cross_val_score(grid_search, data_train_tr, data_train['direction'], cv=5, scoring="accuracy")
```

    array([0.61237785, 0.5228013 , 0.52687296, 0.52117264, 0.52730236])


```python
precision_score(data_train['direction'], y_pred_search)
```

    0.5300402872657208


上面的结果显示，模型的准确率较之前有所提高的。接着，我们将最优的模型保存下来，并应用到测试集上。


```python
grid_search.best_estimator_
```

    LogisticRegression(C=0.001)


```python
final_model = grid_search.best_estimator_
```


```python
y_pred_search = final_model.predict(data_test_tr)
```


```python
accuracy_score(data_test['direction'], y_pred_search)
```

    0.529641693811075
