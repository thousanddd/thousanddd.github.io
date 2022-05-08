---
layout: post
title:  "2.3 时间序列建模：深度学习（3/3）"
date:   2022-05-01
categories: 时间序列 深度学习 RNN
---


根据之前实验，RNN在预测具体价格的时准确率并不高，这期我们接着看下其在预测市场走势、和市场波动率上否有所提高。


```python
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from pprint import pprint
from pylab import plt, mpl
```


导入沪深300指数过去5年的收盘价数据，并拆分成训练集和测试集。


```python
import akshare as ak
ticker = 'sh000300'
csi300_daily = ak.stock_zh_index_daily(symbol=ticker)
split = csi300_daily.shape[0]-220*5
csi300_close = csi300_daily.close.iloc[split:]
csi300_date = csi300_daily.date.iloc[split:]
```


```python
split = int(len(csi300_close.values)*0.90)
csi300_train = csi300_close.iloc[:split]
csi300_test = csi300_close.iloc[split:]
date_train = csi300_date[:split]
date_test = csi300_date[split:]
```


```python
x = date_train
y = csi300_train
plt.figure(figsize=(10, 6))
plt.plot(x,y)
plt.title('CSI 300 Index Daily Close Price - In Sample')
plt.ylabel('Price')
plt.xlabel('Date')
plt.grid()
```

    
![output_5_0.png](https://s2.loli.net/2022/05/01/bLaTRGS4CI6tqYm.png)
    

```python
x = date_test
y = csi300_test
plt.figure(figsize=(10, 6))
plt.plot(x,y)
plt.title('CSI 300 Index Daily Close Price - Out Of Sample')
plt.ylabel('Price')
plt.xlabel('Date')
plt.grid()
```


![output_6_0.png](https://s2.loli.net/2022/05/01/CT1ljXefhur9Zxw.png)
    

```python
csi300_train = pd.DataFrame(csi300_train)
csi300_train = csi300_train.values
csi300_test = pd.DataFrame(csi300_test)
csi300_test = csi300_test.values
```


```python
csi300_train[:5]
```


    array([[3930.798],
           [3959.395],
           [3976.949],
           [3993.575],
           [4021.968]])


```python
csi300_test[len(csi300_test)-5:]
```


    array([[3814.913],
           [3784.12 ],
           [3895.536],
           [3921.107],
           [4016.241]])


```python
from keras.preprocessing.sequence import TimeseriesGenerator
```


```python
lags = 5
```


```python
g_train = TimeseriesGenerator(csi300_train, csi300_train, length=lags, batch_size=128)
g_test = TimeseriesGenerator(csi300_test, csi300_test, length=lags, batch_size=128)
```


```python
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense
```


基于之前的分析结论，我们用LSTM（Long Short-Term Memory）构建RNN模型，并通过样本外的测试集判断预测的准确率。


```python
model = Sequential()
model.add(LSTM(256, activation='relu', input_shape=(lags, 1)))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adagrad', loss='mse', metrics=['mae'])
```


```python
model.summary()
```

    Model: "sequential_19"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     lstm_16 (LSTM)              (None, 256)               264192    
                                                                     
     dense_24 (Dense)            (None, 1)                 257       
                                                                     
    =================================================================
    Total params: 264,449
    Trainable params: 264,449
    Non-trainable params: 0
    _________________________________________________________________
    


```python
%%time
model.fit(g_train, epochs=2000, steps_per_epoch=5, verbose=False)
```

    CPU times: total: 15min 36s
    Wall time: 2min 47s

    <keras.callbacks.History at 0x1ee856e2590>


```python
res = pd.DataFrame(model.history.history)
res.tail(3)
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
      <th>loss</th>
      <th>mae</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1997</th>
      <td>2978.219727</td>
      <td>40.449360</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>2829.699707</td>
      <td>38.647892</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>3196.214600</td>
      <td>42.655376</td>
    </tr>
  </tbody>
</table>
</div>


我们用最近一个交易日前5天的数据来预测最后一个交易日的沪深300的价格。


```python
x_last = g_test[0][0][104]
x_last = x_last.reshape(1, lags, 1)
```


```python
y_pred = model.predict(x_last, verbose=False)
print(y_pred)
```

    [[3900.4976]]
    

```python
y = model.predict(g_test, verbose=False)
y_hat = y.flatten()
x_hat = csi300_test[lags:]
```


```python
fig, ax=plt.subplots(figsize=(10,6))
ax.plot(x_hat[len(x_hat)-10:], label='Actual Stock Price', linestyle='--')
ax.plot(y_hat[len(y_hat)-10:], linestyle='solid', label="Prediction")
ax.set_title('Out Of Sample Prediction Of Stock Price- CSI 300 Index - 2 Weeks')
ax.set(xlabel='Date', ylabel='Closed Price')
ax.legend(loc='best')
plt.grid()
```

    
![output_23_0.png](https://s2.loli.net/2022/05/01/7cS98nZLmQlDMrd.png)
    

完整的样本外预测结果如下


```python
fig, ax=plt.subplots(figsize=(10,6))
ax.plot(x_hat, label='Actual Stock Price', linestyle='--')
ax.plot(y_hat, linestyle='solid', label="Prediction")
ax.set_title('Out Of Sample Prediction Of Stock Price- CSI 300 Index')
ax.set(xlabel='Date', ylabel='Closed Price')
ax.legend(loc='best')
plt.grid()
```

    
![output_25_0.png](https://s2.loli.net/2022/05/01/8vDpSGlWaAKL94B.png)
    

接着，基于预测值计算沪深300指数的变化，并统计预测正确的次数。


```python
act_close = pd.DataFrame(x_hat)
pred_close = pd.DataFrame(y_hat)
```


```python
act_chg = act_close/act_close.shift(1)-1
act_chg.dropna(inplace = True)
act_chg.shape
```


    (104, 1)


```python
pred_chg = pred_close/pred_close.shift(1)-1
pred_chg.dropna(inplace = True)
pred_chg.shape
```


    (104, 1)


```python
from sklearn.metrics import accuracy_score
```


```python
accuracy_score(np.sign(act_chg), np.sign(pred_chg))
```


    0.4519230769230769


我们初步结论的是：当前的RNN模型在预测第二天市场走势的准确率也不高，下面是对波动率的分析。首先是对训练集做预处理，用过去5天的价格变化作为当天的波动率，即Realized Volatility，如下所示。


```python
csi300_train = pd.DataFrame(csi300_train)
```


```python
train_chg = csi300_train/csi300_train.shift(1)-1
train_chg.dropna(inplace = True)
train_realized_vol = train_chg.rolling(5).std()
train_realized_vol.dropna(inplace = True)
```


```python
plt.figure(figsize=(10, 6))
plt.plot(train_realized_vol)
plt.title('Realized Volatility- CSI300 Index')
plt.ylabel('Volatility')
plt.grid()
```

    
![output_35_0.png](https://s2.loli.net/2022/05/01/FtAbDOmRaX9V1vM.png)
    

在将原始数据导入模型前，将Realized Volatility做了标准化处理，并转化成三维数据形式。


```python
train_realized_vol = pd.DataFrame(train_realized_vol)
train_realized_vol = train_realized_vol.values
train_realized_vol = (train_realized_vol-train_realized_vol.mean())/train_realized_vol.std()
```


```python
g_train = TimeseriesGenerator(train_realized_vol, train_realized_vol, length=lags, batch_size=128)
```


```python
model = Sequential()
model.add(LSTM(256, activation='relu', input_shape=(lags, 1)))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adagrad', loss='mse', metrics=['mae'])
```


```python
model.summary()
```

    Model: "sequential_21"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     lstm_18 (LSTM)              (None, 256)               264192    
                                                                     
     dense_26 (Dense)            (None, 1)                 257       
                                                                     
    =================================================================
    Total params: 264,449
    Trainable params: 264,449
    Non-trainable params: 0
    _________________________________________________________________
    

```python
%%time
model.fit(g_train, epochs=2000, steps_per_epoch=5, verbose=False)
```

    CPU times: total: 16min 21s
    Wall time: 2min 50s
    
    <keras.callbacks.History at 0x1ee8bf5d7b0>


```python
y = model.predict(g_train, verbose=False)
y_hat = y.flatten()
x_hat = train_realized_vol[lags:]
```


```python
fig, ax=plt.subplots(figsize=(10,6))
ax.plot(x_hat, label='Realized Volatility', linestyle='--')
ax.plot(y_hat, linestyle='solid', label="Prediction")
ax.set_title('Prediction Of CSI 300 Index Volatility - In Sample')
ax.set(xlabel='Date', ylabel='Volatility')
ax.legend(loc='best')
plt.grid()
```

    
![output_43_0.png](https://s2.loli.net/2022/05/01/1IK3O24ETQfbYGh.png)
    

对测试集也做同样的数据预处理，具体如下。


```python
csi300_test = pd.DataFrame(csi300_test)
test_chg = csi300_test/csi300_test.shift(1)-1
test_chg.dropna(inplace = True)
test_realized_vol = test_chg.rolling(5).std()
test_realized_vol.dropna(inplace = True)
```


```python
plt.figure(figsize=(10, 6))
plt.plot(test_realized_vol)
plt.title('Realized Volatility- CSI300 Index')
plt.ylabel('Volatility')
plt.grid()
```

    
![output_46_0.png](https://s2.loli.net/2022/05/01/Nx6PA8qt1XSHQgK.png)
    

```python
test_realized_vol = pd.DataFrame(test_realized_vol)
test_realized_vol = test_realized_vol.values
test_realized_vol = (test_realized_vol-test_realized_vol.mean())/test_realized_vol.std()
```


```python
g_test = TimeseriesGenerator(test_realized_vol, test_realized_vol, length=lags, batch_size=128)
```


```python
y = model.predict(g_test, verbose=False)
y_hat = y.flatten()
x_hat = test_realized_vol[lags:]
```


```python
fig, ax=plt.subplots(figsize=(10,6))
ax.plot(x_hat, label='Realized Volatility', linestyle='--')
ax.plot(y_hat, linestyle='solid', label="Prediction")
ax.set_title('Prediction Of CSI 300 Index Volatility - Out Of Sample')
ax.set(xlabel='Date', ylabel='Volatility')
ax.legend(loc='best')
plt.grid()
```

    
![output_50_0.png](https://s2.loli.net/2022/05/01/4XsRUWCONn2G8S5.png)
    

从样本外预测结果看，RNN对波动率的预测效果尚可接受。对波动率的预测，可以帮助我们在交易日当天调整仓位，比如，为保证当前组合稳定在一个固定的波动率范围内。
