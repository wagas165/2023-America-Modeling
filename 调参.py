import pandas as pd
import torch
import numpy as np
import scipy.stats as ss

df=pd.read_excel("/Users/zhangyichi/Desktop/识别.xls")
v_num=torch.tensor(df['v_num'])
v_rep_num=torch.tensor(df['v_rep_num'])
c_num=torch.tensor(df['c_num'])
c_rep_num=torch.tensor(df['c_rep_num'])
frequency=torch.tensor(df['sigmoid'])
mls=torch.tensor(df['mls'])
need=[]
for iter in range(100000):
    if iter%1000==0:
        print(f'迭代{iter}次')
    list=[]
    num=float(-np.random.rand(1))
    list.append(num)
    num = float(np.random.rand(1))
    list.append(num)
    num = float(np.random.rand(1))
    list.append(num)
    num = float(np.random.rand(1))
    list.append(num)
    num = float(-np.random.rand(1))
    list.append(num)
    num = float(np.random.rand(1))
    list.append(num)
    num=list[0]*v_num+list[1]*v_rep_num+list[2]*c_num+list[3]*c_rep_num+list[4]*frequency+list[5]*mls
    X=pd.DataFrame(np.array(num))
    X=X.iloc[:,0]
    Y=df['expectancy']
    result2=ss.pearsonr(X,Y)
    if iter==0:
        need.append(result2[0])
        need2=list
    else:
        if result2[0]>need[0]:
            need[0]=result2[0]

print(need,need2)
list=need2
num=list[0]*v_num+list[1]*v_rep_num+list[2]*c_num+list[3]*c_rep_num+list[4]*frequency+list[5]*mls
X=pd.DataFrame(np.array(num))
X.to_excel('/Users/zhangyichi/Desktop/difficulty.xls')