# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:20:56 2023

@author: woute
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.special import loggamma, gamma 
from scipy.stats import t as tl, skewnorm
import random




def skew_t_log_likelihood(params, x):
    alpha, beta, nu, omega, labda = params
    w = labda / np.sqrt(nu)
    logpdf_t = tl.logpdf(x, nu)
    logpdf_skewnorm = skewnorm.logpdf(x,w)
    log_likelihood = np.log((1 - labda) * np.exp(logpdf_t) + labda * np.exp(logpdf_skewnorm))
    return log_likelihood




def distribution_sigma_T(series,params):
    T = len(series)
    sigma2 = np.zeros(T)
    sigma2[0] = np.log(np.var(series))
    for t in range(1, T):
        score = -0.5+0.5*(params['nu']+1)*((series[t-1]**2)/(((params['nu']-2)*np.exp(sigma2[t-1]))+series[t-1]**2))
        sigma2[t] = params['omega'] + params['alpha'] * score + params['beta'] * sigma2[t-1]

    h = np.exp(sigma2)
    return h 

def distribution_sigma_Tskew(series, params):
    T = len(series)
    sigma2 = np.zeros(T)
    sigma2[0] = np.log(np.var(series))
    m = gamma((params['nu']-1)/2)/(gamma(params['nu']/2))*np.sqrt((params['nu']-2)/np.pi)*(params['labda']-1/params['labda'])
    s = np.sqrt((params['labda']**2+(1/params['labda']**2)-1)-m**2)
    for t in range(1, T):
        
        I = (s* (series[t-1]/np.sqrt(np.exp(sigma2[t-1]))))+m
        if I<0:
            I=-1
        else:
            I=1
        score = -0.5+((((s*(series[t-1]/np.exp(0.5*sigma2[t-1])) *((s*series[t-1]/
                np.sqrt(np.exp(sigma2[t-1])))+m)*params['labda']**(-2*I)))/((params['nu']-2)+
                (((s*series[t-1]/np.sqrt(np.exp(sigma2[t-1])))+m)**2 *params['labda']**(-2*I))))
                *((params['nu']+1)/2))
        sigma2[t] = params['omega'] + params['alpha'] * score + params['beta'] * sigma2[t-1]
    h = np.exp(sigma2)
    return h                                                                                                 
 

def calc_sigma2(series, params):
    h = distribution_sigma_T(series,params)
    return h

def log_likelihood(params, series, calc_sigma2):
    alpha, beta, nu, omega = params
    
    T = len(series)
    log_likelihood = 0.0
    sigma2 = calc_sigma2(series, {'alpha': alpha, 'beta': beta , 'nu': nu, 'omega': omega})
    
    for i in range(1, T):
        x = series[i] / np.sqrt(sigma2[i])

        log_likelihood += tl.logpdf(x, nu) - 0.5*np.log(sigma2[i])   

    return -log_likelihood

def log_likelihood_skew(params, series, distribution_sigma_Tskew):
    alpha, beta, nu, omega, labda = params
    
    T = len(series)
    log_likelihood = 0.0
    sigma2 = distribution_sigma_Tskew(series, {'alpha': alpha, 'beta': beta , 'nu': nu, 'omega': omega, 'labda' : labda })
    for i in range(1, T):
        x = series[i] / np.sqrt(sigma2[i])

        log_likelihood += skew_t_log_likelihood(params,x) 

    return -log_likelihood




def fit_gas(series, calc_sigma2):
    # Define initial parameter values and bounds for optimization
    alpha_0 = 0.05
    beta_0 = 0.5
    nu_0 = 5.0
    omega_0 = 10
    initial_params= [alpha_0, beta_0, nu_0, omega_0]
    
    
    bounds = ( (0, 1 ), (0, 1), (3, 100),(-20,100))
    
    # Define the negative log-likelihood as the objective function
    obj_func = lambda params: log_likelihood(params, series, calc_sigma2)
    
    
    A = np.array([[1.0, 1.0, 0, 0]])
    b = np.array([1.0])
    linear_constraint = LinearConstraint(A, lb=-np.inf, ub=b)
    # Use scipy.optimize.minimize to find the maximum of the negative log-likelihood
    result = minimize(obj_func, initial_params, bounds=bounds, constraints=[linear_constraint])
    return result.x



def fit_gas_skew(series, distribution_sigma_Tskew):
    # Define initial parameter values and bounds for optimization
    alpha_0 = 0.05
    beta_0 = 0.5
    nu_0 = 5.0
    omega_0 = 10 
    di_0 = 0.01
    initial_params= [alpha_0, beta_0, nu_0, omega_0,di_0]
    
    
    bounds = ( (0, 1 ), (0, 1), (3, None),(-10,10),(0,100))
    
    # Define the negative log-likelihood as the objective function
    obj_func = lambda params: log_likelihood_skew(params, series, distribution_sigma_Tskew)
    
    
    A = np.array([[1.0, 1.0, 0, 0,0]])
    b = np.array([1.0])
    linear_constraint = LinearConstraint(A, lb=-np.inf, ub=b)
    # Use scipy.optimize.minimize to find the maximum of the negative log-likelihood
    result = minimize(obj_func, initial_params, bounds=bounds, constraints=[linear_constraint])
    return result.x

def monte_carlo_t(params, vardata,simN):    
# Scale the shocks by the square root of the corresponding conditional variance
    sigma2 = np.zeros(simN)
    h = vardata[column].iloc[-1]
    
    # Generate 10 random shocks from a standard normal distribution
    shocks = np.random.standard_t(params['nu'], simN)
    
    scaled_shocks = np.zeros(simN)
    
    scaled_shocks[0] = shocks[0] * np.sqrt(h)
    
    
    sigma2[0] = np.log(h)
    
    total = np.zeros(simN)
    total[0]= df[column].iloc[-1] *  (1+ scaled_shocks[0]/100)
    
    for t in range(1, simN):
        score = -0.5+0.5*(params['nu']+1)*((scaled_shocks[t-1]**2)/(((params['nu']-2)*np.exp(sigma2[t-1]))+scaled_shocks[t-1]**2))
        sigma2[t] = params['omega'] + params['alpha'] * score + params['beta'] * sigma2[t-1]
        
        scaled_shocks[t] = shocks[t]  * np.sqrt(np.exp(sigma2[t]))
        total[t] =  total[t-1] *  (1+ scaled_shocks[t] * (1+ t/100) / 100) 
    
    return total

def normalize_monte(result, vardata,simN):
    total = vardata[:simN]
    total[:] = 0
    total2 = total
    for column in vardata.columns:
        params = results[column]
        total[column] = monte_carlo_t(params, vardata,simN)     
    
    df1 = total.sum(1)
    for i in range(0,simN):
        total2[i:i+1] = round(total[i:i+1] / df1[i] * 150)
        if total2[i:i+1].sum(1)[0] != 150:
            while total2[i:i+1].sum(1)[0] < 150:
                team = random.choice(data.columns)
                total2[team][i] = total2[team][i] + 1 
            while total2[i:i+1].sum(1)[0] > 150:
                team = random.choice(data.columns)
                total2[team][i] = total2[team][i] - 1    


                 
    otalshare[column]

# Define example dataframe with two time series
df = pd.read_excel("peiling.xlsx")
df = df[df.columns.drop(list(df.filter(regex='.low')))]
df = df[df.columns.drop(list(df.filter(regex='.high')))]
df = df[df.index % 1 == 0 ]
df = df.reset_index()
# Take the first difference of each time series
data2 = df.iloc[:, 2:].diff()
data2.index -= 1
data3= df.iloc[:-1, 2:] + data2
data = np.log(data3/df.iloc[:-1, 2:])
data = data.iloc[1:] * 100
vardata = data

simN = 40 


# Estimate the GAS(1,1) model for each series and store the results in a dictionary
results = {}
for column in data.columns:
    series = data[column]
    params= fit_gas(series, calc_sigma2)
    results[column] = {
        'alpha': params[0],
        'beta': params[1],
        'nu': params[2],
        'omega' : params[3]}
    vardata[column] =  calc_sigma2(series, results[column])

    
monte_carlo_t(params, vardata, simN)




#results2 = {}
#for column in data.columns:
#    series = data[column]
#    params= fit_gas_skew(series, distribution_sigma_Tskew)
#    results2[column] = {
#        'alpha': params[0],
#        'beta': params[1],
#        'nu': params[2],
#        'omega' : params[3],
 #       'labda' : params[4],
  #  }

    
    

# Calculate the conditional variances for each series using the estimated parameters



