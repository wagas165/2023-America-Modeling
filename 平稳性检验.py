from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_excel("C:\\Users\\31626\\Desktop\\美赛\\Problem_C_Data_Wordle.xlsx")
dta=df.iloc[:,3]
date=df.iloc[:,0]
diff1=dta.diff(1)
def stableCheck(timeseries):
    # 移动12期的均值和方差
    rol_mean = timeseries.rolling(window=12).mean()
    rol_std = timeseries.rolling(window=12).std()
    # 绘图
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rol_mean, color='red', label='Rolling Mean')
    std = plt.plot(rol_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.savefig("C:\\Users\\31626\\Desktop\\美赛\\pingwendiff1.png")
    # 进行ADF检验
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    # 对检验结果进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print('ADF检验结果:')
    print(dfoutput)

stableCheck_result1 = stableCheck(diff1[1:])