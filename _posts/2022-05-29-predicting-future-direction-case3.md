---
layout: post
title:  "2.8 预测大盘走势（3/3）-深度神经网络模型"
date:   2022-05-29
categories: 大盘走势 深度学习 VIX
---


上一篇文章中，我们通过逻辑回归模型将上证50指数的预测准确率，在测试集上提升了一些。如文章结尾所示，我们继续同样的实验，但换用深度学习模型做预测，具体步骤如下。


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


在这个实验中，我们对指数方向的预测，即$y$变量并没有使用+1或-1，而是将其设定为0或1。因为在后面的DNN模型中，我们将会使用activation函数='sigmoid'。


```python
data['return'] = np.log(data['close']/data['close'].shift(1))
data['direction'] = np.where(data['return'] > 0, 1, 0)
data.dropna(inplace = True)
data.info
```


    <bound method DataFrame.info of          close    return  direction
    1     1060.801  0.047741          1
    2     1075.656  0.013906          1
    3     1086.303  0.009849          1
    4     1102.662  0.014947          1
    5     1081.785 -0.019115          0
    ...        ...       ...        ...
    4464  2782.548 -0.009085          0
    4465  2734.192 -0.017531          0
    4466  2743.417  0.003368          1
    4467  2744.898  0.000540          1
    4468  2763.072  0.006599          1
    
    [4468 rows x 3 columns]>


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


### 拆分训练及测试集


我们仍然用大部分的数据，95%的作为训练集；剩下的5%，也就是最近一年的数据作为测试集。并将两个子集分别做标准化处理。这里需要指出的是，我们用在训练集和测试集上使用同样的的高斯分布。


```python
split = int(len(data)*0.95)
data_train = data[:split]
data_test = data[split:]
```


```python
mu, std = data_train.mean(), data_train.std()
```


```python
data_train_nor = (data_train - mu) / std
```


```python
data_test_nor = (data_test - mu) / std
```


### 训练深度神经网络模型


导入DNN相关的API接口和模型相关参数，如下：


```python
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
```


```python
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(lags,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 64)                384       
                                                                     
     dense_1 (Dense)             (None, 64)                4160      
                                                                     
     dense_2 (Dense)             (None, 1)                 65        
                                                                     
    =================================================================
    Total params: 4,609
    Trainable params: 4,609
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.fit(data_train_nor[cols], data_train['direction'], 
          epochs=50, verbose=False, validation_split=0.2, shuffle=False)
```


    <keras.callbacks.History at 0x15e9f502470>


```python
res = pd.DataFrame(model.history.history)
```


```python
res[['accuracy', 'val_accuracy']].plot(figsize=(10, 6), style='--')
plt.title('DNN分类算法在训练集和验证集上准确率')
plt.ylabel('%')
plt.xlabel('步数')
plt.grid()
```

   
![output_23_0.png](https://s2.loli.net/2022/05/29/QtlxmZ2HGRrpSwb.png)
    

### 训练集上的预测效果


```python
model.evaluate(data_train_nor[cols], data_train['direction'])
```

    133/133 [==============================] - 0s 742us/step - loss: 0.6511 - accuracy: 0.6136
    
    [0.6510671377182007, 0.6135880947113037]



我们选择$cutoff = 0.5$作为阀值，模型的预测结果，即Y大于0.5则代表涨，反之为跌。然后再将Y的值转化为1和-1，这样就和之前机器学习的算法对应上了。


```python
pred = np.where(model.predict(data_train_nor[cols]) > 0.5, 1, 0)
```


```python
data_train['prediction'] = np.where(pred > 0, 1, -1)
```


```python
data_train['prediction'].value_counts()
```


     1    2213
    -1    2026
    Name: prediction, dtype: int64


```python
hits = np.sign(data_train['return'] * data_train['prediction']).value_counts()
```


```python
hits.values[0] / sum(hits)
```


    0.613588110403397


仅从训练集的预测准确率看，DNN的模型效果比之前的两个模型要好。但我们更专注其在测试集上的结果，避免模型在训练集的过度拟合。


### 测试集上的预测效果


```python
model.evaluate(data_test_nor[cols], data_test['direction'])
```


    7/7 [==============================] - 0s 1ms/step - loss: 0.7012 - accuracy: 0.5000  

    [0.7011973261833191, 0.5]


```python
pred = np.where(model.predict(data_test_nor[cols]) > 0.5, 1, 0)
```


```python
data_test['prediction'] = np.where(pred > 0, 1, -1)
```


```python
data_test['prediction'].value_counts()
```


    -1    114
     1    110
    Name: prediction, dtype: int64


```python
hits = np.sign(data_test['return'] * data_test['prediction']).value_counts()
```


```python
hits.values[0] / sum(hits)
```


    0.5


```python
data_test['strategy'] = (data_test['prediction'] * data_test['return'])
```


```python
data_test[['return', 'strategy']].sum().apply(np.exp)
```


    return      0.795967
    strategy    0.757523
    dtype: float64


```python
data_test[['return', 'strategy']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.title('深度神经网络模型预测上证50走势 - 样本外')
plt.ylabel('Rmb')
plt.xlabel('Days')
plt.grid()
```


![output_42_0.png](https://s2.loli.net/2022/05/29/VC3BQZirtujFNvJ.png)
    

从测试集的预测结果和回测数据看，DNN模型对新数据的预测效果并不好。接着，我们尝试增加一些新的输入变量到模型中，看是否能改善测试集的结果。


### 增加新的输入变量


这里的新变量主要是指从日涨跌幅衍生处理的一些变量，包括：波动率，移动平均线。在计算这些衍生变量的时候，需要增加shift(1)的计算，主要是避免将被预测的日涨跌幅提前用到了输入变量中。


```python
data['momentum'] = data['return'].rolling(5).mean().shift(1)
```


```python
data['volatility'] = data['return'].rolling(10).std().shift(1)
```


```python
data.dropna(inplace=True)
```


```python
cols.extend(['momentum', 'volatility'])
```


增加新变量后的输入变量，大概是下面这个样子：


```python
print(data.round(4).tail())
```

             close  return  direction   lag_1   lag_2   lag_3   lag_4   lag_5  \
    4464  2782.548 -0.0091          0  0.0228 -0.0004 -0.0041  0.0133 -0.0102   
    4465  2734.192 -0.0175          0 -0.0091  0.0228 -0.0004 -0.0041  0.0133   
    4466  2743.417  0.0034          1 -0.0175 -0.0091  0.0228 -0.0004 -0.0041   
    4467  2744.898  0.0005          1  0.0034 -0.0175 -0.0091  0.0228 -0.0004   
    4468  2763.072  0.0066          1  0.0005  0.0034 -0.0175 -0.0091  0.0228   
    
          momentum  volatility  
    4464    0.0043      0.0110  
    4465    0.0045      0.0110  
    4466   -0.0016      0.0126  
    4467   -0.0002      0.0123  
    4468    0.0000      0.0120  
    

由于变量发生了变化，我们需要重新拆分训练集和测试集。为了简化文章的篇幅，下面仅验证在测试集上的模型效果。


```python
split = int(len(data)*0.95)
data_train = data[:split]
data_test = data[split:]
```


```python
mu, std = data_train.mean(), data_train.std()
data_train_nor = (data_train - mu) / std
data_test_nor = (data_test - mu) / std
```


```python
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(len(cols),)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```


```python
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_3 (Dense)             (None, 64)                512       
                                                                     
     dense_4 (Dense)             (None, 64)                4160      
                                                                     
     dense_5 (Dense)             (None, 1)                 65        
                                                                     
    =================================================================
    Total params: 4,737
    Trainable params: 4,737
    Non-trainable params: 0
    _________________________________________________________________
    

```python
model.fit(data_train_nor[cols], data_train['direction'], 
          epochs=50, verbose=False, validation_split=0.2, shuffle=False)
```


    <keras.callbacks.History at 0x15eaaae80a0>


```python
model.evaluate(data_train_nor[cols], data_train['direction'])
```

    133/133 [==============================] - 0s 729us/step - loss: 0.6378 - accuracy: 0.6255

    [0.6378472447395325, 0.6255319118499756]


```python
model.evaluate(data_test_nor[cols], data_test['direction'])
```

    7/7 [==============================] - 0s 1ms/step - loss: 0.6885 - accuracy: 0.5785

    [0.6884780526161194, 0.5784753561019897]


```python
pred = np.where(model.predict(data_test_nor[cols]) > 0.5, 1, 0)
```


```python
data_test['prediction'] = np.where(pred > 0, 1, -1)
```


```python
data_test['strategy'] = (data_test['prediction'] * data_test['return'])
```


```python
data_test[['return', 'strategy']].sum().apply(np.exp)
```


    return      0.793095
    strategy    1.122034
    dtype: float64


新的变量将模型准确率提升了一些，对应测试集回测的业绩结果也比被动持有带来了更好的业绩。


```python
data_test[['return', 'strategy']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.title('深度神经网络模型预测上证50走势 - 样本外')
plt.ylabel('Rmb')
plt.xlabel('Days')
plt.grid()
```

 
![output_65_0.png](https://s2.loli.net/2022/05/29/I4HEzDl1gBvyaXV.png)
