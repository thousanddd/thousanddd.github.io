---
layout: post
title:  "2.2 时间序列建模：深度学习（2/3）"
date:   2022-04-30
categories: 时间序列 深度学习 RNN
---


我们继续之前的实验，通过RNN模型预测沪深300的价格。


```python
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from pprint import pprint
from pylab import plt, mpl
```


导入沪深300指数过去5年的收盘价数据。


```python
import akshare as ak
ticker = 'sh000300'
csi300_daily = ak.stock_zh_index_daily(symbol=ticker)
split = csi300_daily.shape[0]-220*5
csi300_close = csi300_daily.close.iloc[split:]
csi300_date = csi300_daily.date.iloc[split:]
```


```python
x = csi300_date
y = csi300_close
plt.figure(figsize=(10, 6))
plt.plot(x,y)
plt.title('CSI 300 Index Daily Close Prices')
plt.ylabel('Rmb')
plt.xlabel('Date')
plt.grid()
```

    
![output_4_0.png](https://s2.loli.net/2022/04/30/RiQ9KCTdxWIU7aj.png)
    

接着，我们将全量数据切分成90%的训练集、和10%的测试集，并将两个样本集都转换为array格式。


```python
split = int(len(csi300_close.values)*0.90)
csi300_train = csi300_close.iloc[:split]
csi300_test = csi300_close.iloc[split:]
```


```python
csi300_train = pd.DataFrame(csi300_train)
csi300_train = csi300_train.values
csi300_test = pd.DataFrame(csi300_test)
csi300_test = csi300_test.values
```


```python
csi300_train.shape
```


    (990, 1)



```python
csi300_test.shape
```


    (110, 1)



再将训练集和测试集转换成RNN需要的三维数据格式，具体操作如下：


```python
from keras.preprocessing.sequence import TimeseriesGenerator
```


```python
lags = 5
```


```python
csi300_train = csi300_train.reshape(len(csi300_train),-1)
csi300_test = csi300_test.reshape(len(csi300_test),-1)
```


```python
g_train = TimeseriesGenerator(csi300_train, csi300_train, length=lags, batch_size=128)
g_test = TimeseriesGenerator(csi300_test, csi300_test, length=lags, batch_size=128)
```


先训练一个SimpleRNN模型。


```python
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense
```


```python
model = Sequential()
model.add(SimpleRNN(200, activation='relu', input_shape=(lags, 1)))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adagrad', loss='mse', metrics=['mae'])
```


```python
model.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     simple_rnn_3 (SimpleRNN)    (None, 200)               40400     
                                                                     
     dense_4 (Dense)             (None, 1)                 201       
                                                                     
    =================================================================
    Total params: 40,601
    Trainable params: 40,601
    Non-trainable params: 0
    _________________________________________________________________
    


```python
%%time
model.fit(g_train, epochs=2000, steps_per_epoch=5, verbose=False)
```

    CPU times: total: 2min 51s
    Wall time: 1min 14s
    

    <keras.callbacks.History at 0x1ca9654c760>


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
      <td>4609.928223</td>
      <td>51.102135</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>4566.030762</td>
      <td>49.494537</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>5143.913086</td>
      <td>52.530804</td>
    </tr>
  </tbody>
</table>
</div>


```python
res.iloc[10:].plot(figsize=(10, 6), style=['--', '--']);
plt.grid()
```

    
![output_21_0.png](https://s2.loli.net/2022/04/30/9EgL1bD5hIuZejd.png)
    

无论从loss function还是mae看，模型收敛的并不好，我们实际看下样本内的预测效果。


```python
y = model.predict(g_train, verbose=False)
```


```python
y_hat = y.flatten()
x_hat = csi300_train[lags:]
```


```python
fig, ax=plt.subplots(figsize=(10,6))
ax.plot(x_hat, label='Actual Stock Price', linestyle='--')
ax.plot(y_hat, linestyle='solid', label="Prediction")
ax.set_title('In Sample Prediction Of Stock Price- CSI 300 Index')
ax.set(xlabel='Date', ylabel='Closed Price')
plt.grid()
```

    
![output_25_0.png](https://s2.loli.net/2022/04/30/ELHsFc7dOJzqNoS.png)
    

初看上面的图，感觉模型效果还是不错的，和之前预判的结果并不相符。我们将上面的图放大一点，比如，随机选择2周数据，即10个点再看看。


```python
fig, ax=plt.subplots(figsize=(10,6))
ax.plot(x_hat[90:100], label='Actual Stock Price', linestyle='--')
ax.plot(y_hat[90:100], linestyle='solid', label="Prediction")
ax.set_title('In Sample Prediction Of Stock Price- CSI 300 Index - 2 Weeks')
ax.set(xlabel='Date', ylabel='Closed Price')
plt.grid()
```

    
![output_27_0.png](https://s2.loli.net/2022/04/30/FJfNXc5YtZo3sbu.png)
    

就具体点位的预测看，误差就很大了，且这还是样本内的。用之前预留的csi300_test这部分干净的数据，再看看样本外的效果。


```python
y = model.predict(g_test, verbose=False)
y_hat = y.flatten()
x_hat = csi300_test[lags:]
```


```python
fig, ax=plt.subplots(figsize=(10,6))
ax.plot(x_hat, label='Actual Stock Price', linestyle='--')
ax.plot(y_hat, linestyle='solid', label="Prediction")
ax.set_title('Out Of Sample Prediction Of Stock Price- CSI 300 Index')
ax.set(xlabel='Date', ylabel='Closed Price')
plt.grid()
```

    
![output_30_0.png](https://s2.loli.net/2022/04/30/QOhZk78gsbJCHIu.png)
    

如上，肉眼可见的效果已经比样本内的差，再近距离看下。


```python
fig, ax=plt.subplots(figsize=(10,6))
ax.plot(x_hat[50:60], label='Actual Stock Price', linestyle='--')
ax.plot(y_hat[50:60], linestyle='solid', label="Prediction")
ax.set_title('Out Of Sample Prediction Of Stock Price- CSI 300 Index - 2 Weeks')
ax.set(xlabel='Date', ylabel='Closed Price')
plt.grid()
```

    
![output_32_0.png](https://s2.loli.net/2022/04/30/uH5CFBPb1ENfq7A.png)
    

我们基本可以得出结论，即：上面训练的RNN模型在实际交易中基本不可用；我们接着用LSTM（Long Short-Term Memory）替换SimpleRNN。


```python
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(lags, 1)))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adagrad', loss='mse', metrics=['mae'])
```


```python
model.summary()
```

    Model: "sequential_7"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     lstm_3 (LSTM)               (None, 200)               161600    
                                                                     
     dense_7 (Dense)             (None, 1)                 201       
                                                                     
    =================================================================
    Total params: 161,801
    Trainable params: 161,801
    Non-trainable params: 0
    _________________________________________________________________
    


```python
%%time
model.fit(g_train, epochs=2000, steps_per_epoch=5, verbose=False)
```

    CPU times: total: 7min 8s
    Wall time: 1min 43s
    

    <keras.callbacks.History at 0x1ca9a396530>


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
      <td>3529.288330</td>
      <td>43.609932</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>3071.022949</td>
      <td>40.291397</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>3085.440186</td>
      <td>41.714394</td>
    </tr>
  </tbody>
</table>
</div>


```python
y = model.predict(g_test, verbose=False)
y_hat = y.flatten()
x_hat = csi300_test[lags:]
```


```python
fig, ax=plt.subplots(figsize=(10,6))
ax.plot(x_hat, label='Actual Stock Price', linestyle='--')
ax.plot(y_hat, linestyle='solid', label="Prediction")
ax.set_title('Out Of Sample Prediction Of Stock Price- CSI 300 Index')
ax.set(xlabel='Date', ylabel='Closed Price')
plt.grid()
```

    
![output_30_0.png](https://s2.loli.net/2022/04/30/QOhZk78gsbJCHIu.png)
    

训练LTSM模型比SimpleNN需要更长的CPU时间，但模型的收敛效果有所提升。


```python
fig, ax=plt.subplots(figsize=(10,6))
ax.plot(x_hat[50:60], label='Actual Stock Price', linestyle='--')
ax.plot(y_hat[50:60], linestyle='solid', label="Prediction")
ax.set_title('Out Of Sample Prediction Of Stock Price- CSI 300 Index - 2 Weeks')
ax.set(xlabel='Date', ylabel='Closed Price')
plt.grid()
```

    
![output_41_0.png](https://s2.loli.net/2022/04/30/DTLWUqMOKSJ9nod.png)
    

放大看样本外预测效果也有所提升，可以在这个模型基础上进一步优化得到更好的预测效果。优化方向包括：对时间序列数据进行预处理，包括：标准化、平划数据里的非平稳性、提升模型的复杂度等。

