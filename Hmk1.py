# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 08:45:12 2021

@author: DariusX
"""
#Cleansing Data
import pandas as pd
import numpy as np 
from numpy import random as normal
import matplotlib.pyplot as plt
from scipy import stats as st
from scipy.stats import studentized_range

df = pd.read_excel (r'C:\Users\DariusX\Desktop\Yale\2022 Asset Management\Quant Investing\Problem_Set1.xlsx',engine='openpyxl',skiprows=4,header=1,)
df= df.drop("Unnamed: 51",axis=1)

#df['date']=pd.to_datetime(df['date'],format=)
#df.info()

#Q1 

#mean_df=pd.DataFrame.mean(df)
#def mean_val():
 #   mean_val=[]
  #  for ind,i in enumerate(('5','10','25','50'),start=0):
   #     mean_val.append(mean_df.iloc[1:int(i)+1].mean())
    #return (mean_val)
#mean_num=pd.DataFrame(mean_val())
#%%
def weights_creation(num_stock):
    w=1/int(num_stock)
    port_weights=[]
    for i in range(1,num_stock+1):
        port_weights.append(w)
    port_weights=np.array(port_weights)
    return port_weights
def mean_returns():
    portfolio=[]
    for i in ('5','10','25','50'):
        portfolio.append(df.iloc[:,1:int(i)+1].multiply(weights_creation(int(i)),axis=1).sum(axis=1))
    portfolio=pd.concat(portfolio,axis=1)
    return portfolio
#%% Mean Returns 
mean_returns()
mean_returns=pd.DataFrame(mean_returns()).rename(columns={0:"5 Stocks", 1:"10 Stocks", 2:"25 Stocks",3: "50 Stocks"})
mean_returns

#%%
def sd_portfolio():
    sd=[]
    for i in ('5','10','25','50'):
        cov_matrix=df.iloc[:,1:int(i)+1].cov()
        sd.append(np.sqrt(np.dot(np.transpose(weights_creation(int(i))),cov_matrix).dot(weights_creation(int(i)))))
    sd=np.array(sd)
    return sd
#%%
sd_portfolio()
stock_num=['5','10','25','50']
plt.plot(stock_num,sd_portfolio())

#%% Q1b).
def variance_part():
    var_contri=[]
    for s_num in ('5','10','25','50'):
        var_val=[]
        for i in range(1,int(s_num)+1):
            var_val.append(df.iloc[:,int(i)].var())       
        var_contri.append(np.dot(var_val,np.transpose(weights_creation(int(s_num))*weights_creation(int(s_num)))))
    return var_contri
    
variance_part()    
var_percent=np.divide((variance_part()),(sd_portfolio())*(sd_portfolio()))
var_percent=np.multiply(var_percent.round(2),100)
plt.plot(stock_num,var_percent)
# Makes sense as stocks increase, no. of covariance terms increase, variance hold lesser contribution
#%% Q1c).
# Higher? Since more weights will be placed on large cap companies, equal weight smooth variances as N increases

#%% Q1d).
st.kstest(mean_returns.iloc[:,int(0)],'norm')
st.kstest(mean_returns.iloc[:,int(0)],'t',(4,))
st.kstest(mean_returns.iloc[:,int(1)],'norm')
st.kstest(mean_returns.iloc[:,int(1)],'t',(9,))
st.kstest(mean_returns.iloc[:,int(2)],'norm')
st.kstest(mean_returns.iloc[:,int(2)],'t',(24,))
st.kstest(mean_returns.iloc[:,int(3)],'norm')
st.kstest(mean_returns.iloc[:,int(3)],'t',(49,))
# Follow normal/ t
def ttest_op():
    stats_output=[]
    pvalues_op=[]
    for i in range(0,4):
        stats_output.append(st.ttest_1samp(mean_returns.iloc[:,int(i)],0))
        pvalues_op.append(stats_output[i].pvalue)
    return pvalues_op

ttest_op()



# Fail to reject null hypothesis(mean=0) for all as p values > 0.05
#%% Q1e).
#Studenized Range
a, b =st.studentized_range(df.CTL, 3, 10)
#Skew
CTL_skew=st.moment(df.CTL, moment=3)
#Kurtosis
CTL_kurt=st.moment(df.CTL, moment=4)
#Normal Test
CTL_normtest=st.normaltest(df.CTL)
np.random.seed(2)
norm_dis=np.random.normal(loc=0,scale=1,size=1000)
norm_skew=st.moment(norm_dis, moment=3)
norm_kurt=st.moment(norm_dis, moment=4)
#CTL is more negatively skewed and exhibit higher kurtosis compared to a standard normal distribution


