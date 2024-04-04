import pmdarima as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

df=pd.read_excel("C:\\Users\\31626\\Desktop\\美赛\\Problem_C_Data_Wordle.xlsx")
time_series=df.iloc[:,3]
date=df.iloc[:,0]

split_point = int(len(time_series) * 0.85)
# 确定训练集/测试集
data_train, data_test = time_series[0:split_point], time_series[split_point:len(time_series)]
# 使用训练集的数据来拟合模型
built_arimamodel = pm.auto_arima(data_train,
                                 start_p=0,
                                 start_q=0,
                                 max_p=5,
                                 max_q=5,
                                 m=15,
                                 d=None,
                                 seasonal=True, start_P=0, D=1, trace=True,
                                 error_action='ignore', suppress_warnings=True,
                                 )

print(built_arimamodel)
# built_arimamodel=sm.tsa.ARIMA(time_series,order=(4,0,4)).fit()
# pred_list = []
# for index, row in data_test.iteritems():
#     # 输出索引，值
#     if index<358:
#         pred_list += [built_arimamodel.predict(n_periods=1)]
#         # 更新模型，model.update()函数，不断用新观测到的 value 更新模型
#         built_arimamodel.update(row)
#         # 预测时间序列以外未来的一次
#         predict_f1 = built_arimamodel.predict(n_periods=1)
#         print('未来一期的预测需求为：', predict_f1[0])
#     else:
#         pred_list += [built_arimamodel.predict(n_periods=1)]
#         print('未来一期的预测需求为：', predict_f1[0])
#
# def forecast_accuracy(forecast, actual):
#     mape = np.mean(np.abs(forecast - actual)/np.abs(actual))
#     me = np.mean(forecast - actual)
#     mae = np.mean(np.abs(forecast - actual))
#     mpe = np.mean((forecast - actual)/actual)
#     # rmse = np.mean((forecast - actual)**2)**0.5    # RMSE
#     rmse_1 = np.sqrt(sum((forecast - actual) ** 2) / actual.size)
#     corr = np.corrcoef(forecast, actual)[0, 1]
#     mins = np.amin(np.hstack([forecast[:, None], actual[:, None]]), axis=1)
#     maxs = np.amax(np.hstack([forecast[:, None], actual[:, None]]), axis=1)
#     minmax = 1 - np.mean(mins/maxs)
#     return ({'mape': mape,
#              'me': me,
#              'mae': mae,
#              'mpe': mpe,
#              'rmse': rmse_1,
#              'corr': corr,
#              'minmax': minmax
#              })
# test_predict = data_test.copy()
# for x in range(len(test_predict)):
#     test_predict.iloc[x] = pred_list[x]
# # 模型评价
# eval_result = forecast_accuracy(test_predict.values, data_test.values)
# print('模型评价结果\n', eval_result)
# forecast=built_arimamodel.predict(60)
#
# # 画图显示
# plt.plot(time_series, 'b-', label='True Data')
# plt.plot(test_predict, 'r-', label='Prediction')
# plt.plot(range(359,419),forecast,'g-', label='Prediction')
# plt.title('RMSE:{}'.format(eval_result['rmse']))
# plt.legend(loc='best')
# plt.grid()  # 生成网格
# plt.show()
