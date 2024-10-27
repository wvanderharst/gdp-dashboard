# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 22:37:25 2022

@author: woute
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import statsmodels.api as sm

df = pd.read_excel("peiling.xlsx")
df = df[df.columns.drop(list(df.filter(regex='.low')))]
df = df[df.columns.drop(list(df.filter(regex='.high')))]
df = df[df.index % 5 == 0 ]
df = df.reset_index()


names = list(df)
names2 = names[2:]

sigmas = df
df = df*100


for i in names2:
    sigmas[i] = np.var(df[i][:3])


N = len(df)
alfa = 0.02
beta = 0.97


name = "VVD"
N = len(df)

def calc_garchparties(theta,names2,sigmas,df,N):
    alfa = theta[0]
    beta = theta[1]
    omega = theta[2]
    llf = 0 
    for j in range(0,17):
        name = names2[j]
        for i in range(1,N):
            error = (df[name][i] -  df[name][i-1])
            sigmas[name][i] = alfa * error**2 + beta * sigmas[name][i-1] 
            #mu[i] = mu[i-1] + np.random.normal(0,sigma2[i])
            llf = llf - np.log(sigmas[name][i]) - error**2/sigmas[name][i]
    print(theta, -llf)
    return -llf 


def calc_ewmaparties(theta,names,sigmas,df,N):
    alfa = theta[0]
    llf = 0 
    for j in range(0,17):
        name = names[j]
        for i in range(1,N):
            error = (df[name][i] -  df[name][i-1])
            sigmas[name][i] = alfa * error**2 + (1-alfa) * sigmas[name][i-1] 
            #mu[i] = mu[i-1] + np.random.normal(0,sigma2[i])
            llf = llf - np.log(sigmas[name][i]) - error**2/sigmas[name][i]
    print(theta, -llf)
    return -llf



def pred_garchparties(theta,name,sigmas,df,N):
    alfa = theta[0]
    beta = theta[1]
    llf = 0 
    for i in range(1,N):
        error =np.random.normal(0,sigma2[i])
        sigmas[name][i] = alfa * error**2 + beta * sigmas[name][i-1] 
        llf = llf - np.log(sigmas[name][i]) - error**2/sigmas[name][i]
    print(theta, -llf)
    return -llf 




theta = [0.07,0.93,0]

linear_constraint = LinearConstraint([[1,1,0],[1,0,0],[0,1,0],[0,0,1]],[0,0,0,0],[1,1,1,100])

minimize(calc_garchparties, theta,args=(names2,sigmas,df,N),constraints = linear_constraint)


sigmas2 = sigmas 
theta = [0.07]
linear_constraint = LinearConstraint([[1]],[0],[1])

minimize(calc_ewmaparties, theta,args=(names2,sigmas,df,N),constraints = linear_constraint)