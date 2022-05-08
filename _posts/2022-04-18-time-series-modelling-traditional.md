---
layout: post
title:  "1.2 时间序列建模：传统方法"
date:   2022-04-18
categories: 时间序列
---


### 移动平均模型


大部分炒股的人对Moving Average都不陌生，我们在任意炒股软件上看股票行情的时候，基本都默认有MA的趋势曲线。移动平均模型可以被理解为，通过对时间序列残值的滞后进行平滑处理形成的模型。这个表述比较拗口，我们用具体的例子来实践下。


根据之前的分析，大概的步骤是这样的。首先，我们把个股的收盘价数据下载下来，去掉缺失值，然后判断该收盘价曲线在平稳性分析上是否显著。接着，将所有的数据切分成训练集和测试集。


```python
sz000001Close=sz000001Close.dropna()
statTest=adfuller(sz000001Close)[0:2]
print("The test statistic and p-value of ADF test are {}"
      .format(statTest))
```

    The test statistic and p-value of ADF test are (-1.281067068919137, 0.6377890306118901)


```python
diff_sz000001Close=sz000001Close.diff().dropna()
split=int(len(diff_sz000001Close.values)*0.9)
diff_train_sz000001=diff_sz000001Close.iloc[:split]
diff_test_sz000001=diff_sz000001Close.iloc[split:]
sm.graphics.tsa.plot_acf(diff_train_sz000001,lags=30)
plt.xlabel('ACF - SZ000001')
plt.show()
```


![output_23_0.png](https://s2.loli.net/2022/04/18/ekbglusSzvjZmQY.png)


上篇文章我们谈到，找到合适的滞后值lag很重要。通过上面这个图，根据超过蓝色置信区间的垂直线，我们就可以选择比较合适的滞后值，分别为short-term=5,long-term=12。


```python
shortMA_sz000001=diff_train_sz000001.rolling(window=5).mean()
longMA_sz000001=diff_train_sz000001.rolling(window=12).mean()
```


```python
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(diff_train_sz000001,
        label='Stock Price', linestyle='--')
ax.plot(shortMA_sz000001,
        label = 'Short MA', linestyle='solid')
ax.plot(longMA_sz000001,
        label = 'Long MA', linestyle='solid')
ax.legend(loc='best')
ax.set_ylabel('Differenced Price')
ax.set_title('Stock Prediction-SZ000001')
plt.show()
```


![output_26_0.png](https://s2.loli.net/2022/04/18/FKLiZbIw26Vc9Jd.png)


从上面画图的结果可以看出，移动平均的窗口设为5的时候，模拟出的移动平均值更敏感，更接近真实的价差（当天-上一天）变化的趋势。而窗口设为12的时候，模拟出来的曲线更加平滑。但通过移动平均并不能达到预测的效果，Autocroorelation Function（ACF）的作图也是为了选择出较合适的平移窗口。<u>对我们的启发是，炒股软件上的MA功能是通用的模板，更科学的做法是基于个股的差异性找到最合适的平移窗口的设置</u>。这里我们就引出下一个模型：自回归模型。


### 自回归模型


自回归模型，即Autoregressive Model，是统计上一种处理时间序列的方法，用同一变数例如x的之前各期，亦即x1至xt-1来预测本期xt的表现，并假设它们为线性关系。因为这是从回归分析中的线性回归发展而来，只是不用x预测y，而是用x预测 x（自己）；所以叫做自回归。和我们经常使用的回归模型Regression Model相比，后者是研究一个变量（被解释变量）关于另一个（些）变量（解释变量）的依赖关系的方法。


自回归模型的优点是它需要的解释变量不多，因为它可用自身变数数列来进行预测。但是这种方法受到一定的限制：必须具有自相关。如果自相关系数(R)小于0.5，则不宜采用，否则预测结果极不准确。从前面我们对平安银行股价的现价$x_t$和前序价格$x_(t-h)$的分析看，有显著相关性的。


在AR模型中我们用Partial Autocorrelation Function，偏自相关函数去找到最合适的平移窗口。同时，为保证时间序列的平稳性，我们仍然使用的是差值，即：当前值和其前一个值的差。从下面画图的结果看，我们选择$lag=5$作为我们的最优窗口值。


```python
sm.graphics.tsa.plot_pacf(diff_train_sz000001, lags=30)
plt.title('PACF of SZ000001 Differed')
plt.xlabel('Number of Lags')
plt.show()
```


![output_30_0.png](https://s2.loli.net/2022/04/18/QEfYtlaOSgGmFzy.png)


```python
from statsmodels.tsa.ar_model import AutoReg
import warnings
warnings.filterwarnings('ignore')
```


```python
ar_sz000001=AutoReg(diff_train_sz000001.values, lags=30)
ar_fitted_sz000001=ar_sz000001.fit()
```


```python
ar_predictions_sz000001=ar_fitted_sz000001.predict(start=len(diff_train_sz000001), 
                                             end=len(diff_train_sz000001)\
                                             + len(diff_test_sz000001) - 1, 
                                             dynamic=False)
```


```python
i=0
for i in range(len(ar_predictions_sz000001)):
    print('==' * 25)
    print('predicted values:{:.4f} & actual values:{:.4f}'\
          .format(ar_predictions_sz000001[i], diff_test_sz000001[split+i+1]))
```

    ==================================================
    predicted values:0.0453 & actual values:0.3300
    ==================================================
    predicted values:0.0941 & actual values:-0.4100
    ==================================================
    predicted values:0.1480 & actual values:-0.8100
    ==================================================
    predicted values:0.1028 & actual values:0.6300
    ==================================================
    predicted values:-0.0625 & actual values:0.1400
    ==================================================
    predicted values:0.0454 & actual values:0.2500
    ==================================================
    predicted values:-0.0676 & actual values:-0.1100
    ==================================================
    predicted values:-0.0826 & actual values:0.5900
    ==================================================
    predicted values:-0.0388 & actual values:-0.1800
    ==================================================
    predicted values:-0.0187 & actual values:0.2000
    ==================================================
    predicted values:-0.0928 & actual values:-0.2200
    ==================================================
    predicted values:-0.0627 & actual values:-0.1300
    ==================================================
    predicted values:-0.0772 & actual values:-0.1700
    ==================================================
    predicted values:-0.0653 & actual values:0.5300
    ==================================================
    predicted values:-0.1038 & actual values:0.1700
    ==================================================
    predicted values:-0.1983 & actual values:0.3700
    ==================================================
    predicted values:0.0545 & actual values:0.6400
    ==================================================
    predicted values:-0.1285 & actual values:-0.1100
    ==================================================
    predicted values:-0.0067 & actual values:0.1200
    ==================================================
    predicted values:-0.0312 & actual values:-0.3500
    ==================================================
    predicted values:-0.0734 & actual values:-0.1300
    ==================================================
    predicted values:-0.0315 & actual values:-0.1200
    ==================================================
    predicted values:-0.0695 & actual values:0.2400
    ==================================================
    predicted values:0.0578 & actual values:0.3800



```python
ar_predictions_sz000001=pd.DataFrame(ar_predictions_sz000001)
ar_predictions_sz000001.index=diff_test_sz000001.index
```


```python
fig, ax=plt.subplots(figsize=(10, 6))
ax.plot(diff_test_sz000001,
        label='Stock Price', linestyle='--')
ax.plot(ar_predictions_sz000001,
        label='Prediction', linestyle='solid')
ax.legend(loc='best')
ax.set_ylabel('Differenced Price')
ax.set_title('Stock Prediction-SZ000001')
plt.show()
```


![output_36_0.png](https://s2.loli.net/2022/04/18/hc9IovHpDYl3Xjs.png)


无论从之前的计算结果和可视化图表都可以看出，即使我们对时间序列做了平稳性调整后，通过AR做出的预测模型仍然是不可用的。接下来我们要继续尝试其它经典的模型，比如下面的：ARIMA模型（Autoregressive Integrated Moving Average model），差分整合移动平均自回归模型。


### 差分整合移动平均自回归模型


ARIMA模型时通过时间序列的历史值和其白噪音整合形成的模型。它可以被看作之前AR和MA模型的通用版本，只是后面两个模型适用的前提条件是平衡性。所以之前建模的时候，我们都是取了股票收盘价的变化值，并没有直接使用原始值。但在ARIMA模型中，我们可以通过对参数$d$的定义，而直接将股票收盘价的时间序列喂给模型训练。


事实上，在ARIMA模型的训练中，有3个核心参数需要设定，分别是$p$，$q$和$d$。前面两个参数分别是之前AR和MA模型训练时，我们用到的参数。参数$d$，也就是当差分阶数$d=0$时，ARIMA模型就等同于ARMA模型。即这两种模型的差别就是差分阶数d是否等于零，也就是时间序列是否平稳，ARIMA模型对应着非平稳时间序列， ARMA模型对应着平稳时间序列。为了能够直接处理原始时间序列，我们设定差分阶数$d=1$。下面我们通过对同一个股票的ARIMA模型训练，实际比较下和之前模型在准确率上差别。


```python
from statsmodels.tsa.arima.model import ARIMA
```


```python
split=int(len(sz000001Close.values)*0.9)
train_sz000001=sz000001Close.iloc[:split]
test_sz000001=sz000001Close.iloc[split:]
```


```python
arima_sz000001=ARIMA(train_sz000001,order=(9,1,9))
arima_fit_sz000001=arima_sz000001.fit()
```


```python
arima_predict_sz000001=arima_fit_sz000001.predict(start=len(train_sz000001), 
                                                  end=len(train_sz000001)\
                                                  + len(test_sz000001) - 1, 
                                                  dynamic=False)
```


```python
arima_predict_sz000001=pd.DataFrame(arima_predict_sz000001)
arima_predict_sz000001.index=test_sz000001.index
```


```python
fig, ax=plt.subplots(figsize=(10, 6))
ax.plot(test_sz000001,
        label='Stock Price', linestyle='--')
ax.plot(arima_predict_sz000001,
        label='Prediction', linestyle='solid')
ax.legend(loc='best')
ax.set_ylabel('Differenced Price')
ax.set_title('Stock Prediction-SZ000001')
plt.show()
```


![output_44_0.png](https://s2.loli.net/2022/04/18/b5Dg4mVsIAaBnvN.png)


很显然，ARIMA模型并没有提升预测的准确性，其实涉及$p$，$q$和$d$3个参数的选择也是缺少理论依据的。找到最优的lag，即：滞后算子本身也不容易、且我们暂时也没找到特别好的方法，只能通过不断试不同的组合，找到最低的AIC值来决定最优的滞后算子。


```python
import itertools
```


```python
p=q= range(0, 9)
d=range(0, 3)
pdq=list(itertools.product(p,d,q))
arima_results_sz000001=[]
for param_set in pdq:
    try:
        arima_sz000001=ARIMA(train_sz000001,order=param_set)
        arima_fitted_sz000001=arima_sz000001.fit()
        arima_results_sz000001.append(arima_fitted_sz000001.aic)
    except:
        continue
print('**'*25)
print('The Lowest AIC score is {:.4f} and the corresponding parameters are {}'
      .format(pd.DataFrame(arima_results_sz000001)
             .where(pd.DataFrame(arima_results_sz000001).T.notnull().all()).min()[0], 
             pdq[arima_results_sz000001.index(min(arima_results_sz000001))]))
```

    **************************************************
    The Lowest AIC score is 247.1340 and the corresponding parameters are (2, 1, 2)


当我们将新的参数组合放到模型中重新顺利后，我们并没有发现模型的预测效果有显著的提升。具体的结果就不这这边展示了。
