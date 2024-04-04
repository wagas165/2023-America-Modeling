from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import  stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot


df=pd.read_excel("C:\\Users\\31626\\Desktop\\美赛\\Problem_C_Data_Wordle.xlsx")
dta=df.iloc[:,3]

decomposition=sm.tsa.seasonal_decompose(dta,model='add',period=15)

trend=decomposition.trend
seasonal=decomposition.seasonal




