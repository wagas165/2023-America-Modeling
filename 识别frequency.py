import pandas as pd

df=pd.read_excel("C:\\Users\\31626\\Desktop\\美赛\\pvalue.xlsx")
df2=pd.read_csv("C:\\Users\\31626\\Desktop\\美赛\\ngram_freq_dict.csv")
lens=len(df)
filt=df[df['Word'].isin(df2.iloc[:,0])]
for word in filt.iloc[:,3]:
    li=[word]
    df=pd.concat([df,df2[df2.iloc[:,0].isin(li)]])
df.to_excel("C:\\Users\\31626\\Desktop\\美赛\\识别.xlsx")
