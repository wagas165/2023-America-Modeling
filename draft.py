import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

#
# with open("C:\\Users\\31626\\Desktop\\美赛\\zlsnb(1).txt",'r',encoding='utf-8') as f:
#     data=f.readlines()
#     dict={}
#     for j in range(1,len(data)):
#         try:
#             i=data[j]
#             num1 = int(i[-15])
#             num2 = int(i[-11])
#             num3 = int(i[-9])
#             num4 = int(i[-5])
#             num5 = int(i[-3:-1])
#             list = [num1,num2,num3,num4,num5]
#             dict[j]=list
#         except:
#             pass
#     df=pd.DataFrame(dict)
#     df.to_excel("C:\\Users\\31626\\Desktop\\美赛\\draft.xlsx")
# df=pd.read_excel("C:\\Users\\31626\\Desktop\\美赛\\pvalue.xlsx")
# data=sns.load_dataset('iris')
df=pd.read_excel('/Users/zhangyichi/Desktop/副本识别.xlsx')
X=df['human expectancy']
Y=df['machine expectancy']
Z=df['eree']
use=pd.concat([X,Y,Z],axis=1)
print(use)
# X=df['difficulty']
# Y=df['percentage']
# result=X.corr(Y)
result2=ss.pearsonr(X,Z)
print(result2)

ax=sns.heatmap(use.corr(),vmax=1,cmap='RdYlGn',annot=True)
plt.show()

