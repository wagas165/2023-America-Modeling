from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import  stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot


df=pd.read_excel("C:\\Users\\31626\\Desktop\\美赛\\Problem_C_Data_Wordle.xlsx")
dta=df.iloc[:,3]
date=df.iloc[:,0]
df2=pd.concat([pd.to_datetime(date),dta],axis=1)
diff1=dta.diff(1)
diff2=diff1.diff(1)
fig=plt.figure()
fig=sm.graphics.tsa.plot_acf(diff1[1:],lags=40)
fig=sm.graphics.tsa.plot_pacf(diff1[1:],lags=40)
fig.show()
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf

#给出最优参数
# def testStationarity(timeSer):
#
#     stationarity = False
#
#     dftest = adfuller(timeSer)
#     dfoutput = pd.Series(dftest[:4], index=[
#                       'Test Statistic', 'p-value', 'lags', 'nobs'])
#
#     for key, value in dftest[4].items():
#         dfoutput['Critical values (%s)' % key] = value
#
#     if dfoutput['Test Statistic'] < dfoutput['Critical values (5%)']:
#         stationarity=dfoutput['p-value']
#
#     return stationarity
# print(testStationarity(dta))
# print(testStationarity(diff1[1:]))
# print(testStationarity(diff2[2:]))
# def p_q_choice(timeSer, nlags=40, alpha=.05):
#
#     kwargs = {'nlags': nlags, 'alpha': alpha}
#     acf_x, confint = acf(timeSer, **kwargs)
#     acf_px, confint2 = pacf(timeSer, **kwargs)
#
#     confint = confint - confint.mean(1)[:, None]
#     confint2 = confint2 - confint2.mean(1)[:, None]
#
#     for key1, x, y, z in zip(range(nlags), acf_x, confint[:,0], confint[:,1]):
#         if x > y and x < z:
#             q = key1
#             break
#
#     for key2, x, y, z in zip(range(nlags), acf_px, confint2[:,0], confint[:,1]):
#         if x > y and x < z:
#             p = key2
#             break
#
#     return p, q
#
# print(p_q_choice(dta),p_q_choice(diff1[1:]),p_q_choice(diff2[2:]))
# from pmdarima.arima import auto_arima
# import joblib
#
# model=auto_arima(dta,start_p=1,start_q=1,max_p=10,max_q=10,trace=True,stepwise=False)
# model.fit(dta)

#常规ARIMA训练模型
# arma_model=sm.tsa.ARIMA(list(dta),order=(4,1,1)).fit()
# predict=arma_model.predict(360,599,dynamic=True)
# print(predict)
# plt.figure()
# plt.plot(range(1,360),arma_model.fittedvalues,color='blue')
# plt.plot(range(360,600),predict,color='green')
# plt.show()

#周期性
decomposition=sm.tsa.seasonal_decompose(dta,model='add',period=15)
decomposition.plot()
plt.show()
trend=decomposition.trend
seasonal=decomposition.seasonal
print(trend)
list=[trend,seasonal]
col=['trend','season']
df=pd.DataFrame(columns=col,data=list)
df.to_excel("C:\\Users\\31626\\Desktop\\美赛\\data.xlsx")
# 引用库函数

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import optimize as op
#
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
# plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
#
# # 需要拟合的函数
# def f_1(x, A, B, C):
#     return A * x**2 + B * x + C
#
# # 需要拟合的数据组
# x_group = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# y_group = [2.83, 9.53, 14.52, 21.57, 38.26, 53.92, 73.15, 101.56, 129.54, 169.75, 207.59]
#
# # 得到返回的A，B值
# A, B, C = op.curve_fit(f_1, x_group, y_group)[0]
#
# # 数据点与原先的进行画图比较
# plt.scatter(x_group, y_group, marker='o',label='真实值')
# x = np.arange(0, 15, 0.01)
# y = A * x**2 + B *x + C
# plt.plot(x, y,color='red',label='拟合曲线')
# plt.legend() # 显示label
#
# plt.show()
