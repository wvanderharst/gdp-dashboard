# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 22:37:25 2022

@author: woute
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint


df = pd.read_excel("peiling.xlsx")
df = df[df.columns.drop(list(df.filter(regex='.low')))]
df = df[df.columns.drop(list(df.filter(regex='.high')))]
df = df[df.index % 14 == 0 ]
df = df.reset_index()


names = list(df)

sigmas = df
sigmas[names] = 0.0

df = pd.read_excel("peiling.xlsx")
df = df[df.columns.drop(list(df.filter(regex='.low')))]
df = df[df.columns.drop(list(df.filter(regex='.high')))]
df = df[df.index % 14 == 0 ]
df = df.reset_index()
df = df*10

N = len(df)
alfa = 0.02
beta = 0.97


name = "VVD"
N = 44

def calc_garchparties(theta,names,sigmas,df,N):
    alfa = theta[0]
    beta = theta[1]
    llf = 0 
    for j in range(0,17):
        name = names[j]
        for i in range(1,N):
            error = df[name][i] -  df[name][i-1]
            sigmas[name][i] = alfa * error**2 + beta * sigmas[name][i-1] 
            #mu[i] = mu[i-1] + np.random.normal(0,sigma2[i])
            llf = llf - np.log(sigmas[name][i]**0,5) - error**2/sigmas[name][i]
        print(theta, -llf)
    return -llf * 10000

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




theta = [0.02,0.97]
names2 = names[2:]


linear_constraint = LinearConstraint([[1, 1],[1,0],[0,1]],[0,0,0],[1,1,1])

minimize(calc_garchparties, theta,args=(names2,sigmas,df,N),constraints = linear_constraint)

    
    