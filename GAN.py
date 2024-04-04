import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import random
import numpy as np
np.set_printoptions(suppress=True)
df=pd.read_excel("/Users/zhangyichi/Library/Mobile Documents/com~apple~CloudDocs/美赛/Problem_C_Data_Wordle.xlsx")

#真实概率分布的切片
def generate_real():
    i=random.randint(1,358)
    real_data=torch.FloatTensor(df.iloc[i,6:13])
    return real_data

#生成鉴别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model=nn.Sequential(
            nn.Linear(7,3),
            nn.Sigmoid(),
            nn.Linear(3,1),
            nn.Sigmoid()
        )

        #loss function
        self.loss_function=nn.MSELoss()

        #optimiser
        self.optimiser=torch.optim.SGD(self.parameters(),lr=0.01)

        self.counter=0
        self.progress=[]
        pass

    def forward(self,inputs):
        return self.model(inputs)

    def train(self,inputs,targets):
        outputs=self.forward(inputs)
        loss=self.loss_function(outputs,targets)

        self.counter+=1
        if self.counter%10==0:
           self.progress.append(loss.item())
           pass
        if self.counter%10000==0:
            print(f'counter={self.counter}')
            pass
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass

    def plot_progress(self):
        df1=pd.DataFrame(self.progress,columns=['loss'])
        df1.plot(alpha=0.1,marker='.',grid=True)
        plt.show()
        pass

#随机生成数据
def generate_random(size):
    random_data=torch.rand(size)
    return random_data

#生成生成器
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model=nn.Sequential(
            nn.Linear(1,3),
            nn.LeakyReLU(),
            nn.Linear(3,7),
            nn.LeakyReLU()
        )
        self.optimiser=torch.optim.Adam(self.parameters(),lr=0.01)
        self.counter = 0
        self.progress = []
        pass

    def forward(self,inputs):
        return self.model(inputs)

    def train(self,D,inputs,targets):
        g_output=self.forward(inputs)
        d_output=D.forward(g_output)
        loss=D.loss_function(d_output,targets)

        self.counter+=1
        if self.counter%10==0:
           self.progress.append(loss.item())
           pass

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass

    def output(self):
        print(self.model)

def generate_randomseed(size):
    random_data=torch.randn(size)
    return random_data

#gan网络构建
D = Discriminator ()
G = Generator ()
for i in range (100000):
    #鉴别器训练真实数据
    D.train (generate_real() , torch.FloatTensor ([1.0]))
    #鉴别器训练虚假数据
    D.train (G.forward (generate_randomseed(1)).detach () , torch.FloatTensor ([0.0]))
    #生成器训练鉴别器结果
    G.train (D,torch.FloatTensor ([0.5]) , torch.FloatTensor ([1.0]))
    pass

#训练已经完成 利用生成器生成数据从而预测
df2=pd.DataFrame()
for i in range(255):
    j=generate_randomseed(1)
    out = G.forward(j)
    out2 = out.detach().numpy()
    #out2=100*out2/np.sum(out2)

    out2 = pd.DataFrame(out2)
    df2 = pd.concat([df2, out2], axis=1)


df2.to_excel("/Users/zhangyichi/Desktop/predict.xls")



