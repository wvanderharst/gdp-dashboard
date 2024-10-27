# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 22:28:33 2022

@author: woute
"""


import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from datetime import datetime
import time 

def recentdata():
    now = datetime.now() # current date and time
    date_time = now.strftime("%m/%d/%Y")
    date_time = date_time.replace('/', '')

    folder = "C:\Tutorial\down\omega" + date_time

    options = webdriver.ChromeOptions()
    prefs = {"download.default_directory" : folder }
    options.add_experimental_option("prefs",prefs)


    driver = webdriver.Chrome(executable_path = "chromedriver",options = options)


    url1 = "https://peilingwijzer.tomlouwerse.nl/resources/Results_Longitudinal.xlsx"

    driver.get(url1)
    time.sleep(5)
    data = pd.read_excel(folder + "\Results_Longitudinal.xlsx")
    driver.quit()
    
    
    return data 




df = recentdata()
df = df[df.columns.drop(list(df.filter(regex='.low')))]
df = df[df.columns.drop(list(df.filter(regex='.high')))]
dataframe = df

dataframe2 = dataframe.iloc[-1][1:]


df = pd.read_excel("politiek.xlsx")


df = df.merge(dataframe2, left_on='Naam',right_index=(True))
df["Zetels peiling "] = df[df.columns[-1]]
df = df.drop(columns=[df.columns[-1]])


table = pd.DataFrame()
table = pd.DataFrame(0.0,columns= df["Naam"],index=df["Naam"])
list(df.columns.values) 

def tablecreator(table,df):
    cols = list(df.columns.values) 
    for k in range(3,len(cols),2):
        cols1 = cols[k]
        cols2 = cols[k+1]
        for i in range(0,len(df)):
            party1 = df["Naam"][i]
            for j in range(0,len(df)):
                party2 = df["Naam"][j]
                table[party1][party2] +=  ((df[cols1][i] -
                                           df[cols1][j])**2 +
                                          (df[cols2][i] - 
                                           df[cols2][j])**2)**0.5
        
    return table

def tablecreator2(df):
    cols = list(df.columns.values) 
    table3 = pd.DataFrame(0.0,columns= df["Naam"],index=df["Naam"])
    for k in range(3,len(cols),2):
        cols1 = cols[k]
        cols2 = cols[k+1]
        table2 = pd.DataFrame(0.0,columns= df["Naam"],index=df["Naam"])

        for i in range(0,len(df)):
            party1 = df["Naam"][i]
            for j in range(0,len(df)):
                party2 = df["Naam"][j]
                table2[party1][party2] =  ((df[cols1][i] -
                                           df[cols1][j])**2 +
                                          (df[cols2][i] - 
                                           df[cols2][j])**2)**0.5
                
            table3[party1] += list(map({j: i for i, j in enumerate(sorted(set(table2[party1])))}.get, table2[party1]))

        
    return table3


allparties = set(df["Naam"])


def checker(N1, N2):
    N3 = N1 / N2 
    if N3 > 2:
        CAT = 2
    elif N3 < 1/2:
        CAT = 2
    elif N3 > 1.7:
        CAT = 1.7
    elif N3 < 1/1.7:
        CAT = 1.7   
    elif N3 > 1.5:
        CAT = 1.5
    elif N3 < 1/1.5:
        CAT = 1.5
    elif N3 > 1.2:
        CAT = 1.2
    elif N3 < 1/1.2:
        CAT = 1.2
    else:
        CAT = 1
    return CAT

def categorizevariance(name,df):
        N1 =  df[name].iloc[-1] 
        N2 =  df[name].iloc[-100]
        N3 =  df[name].iloc[-200] 
        N4 =  df[name].iloc[-400]
        return max(checker(N1, N2),checker(N1, N3),checker(N1, N4),checker(N2, N3),checker(N2, N4),checker(N3, N4))
       
def montecarloelection(df,sigma,dataframe):
    df2 = df["Zetels peiling "]
    for i in range(0,len((df2))):
        name = df["Naam"][i]
        CAT = categorizevariance(name,dataframe)
        df2[i] = df2[i] + np.random.normal(0,( 0.15 * CAT**2  * df2[i])) 
        if df2[i] < 0:
            df2[i] = 0
    data = 150 * df2/sum(df2)
    for i in range(0,len((df2))):
        df2[i] = round(data[i])
        
    while sum(df2) < 150:
        i= np.random.randint(0, len(df2))
        df2[i] += 1
    while sum(df2) > 150:
        i= np.random.randint(0, len(df2))
        if df2[i]>1:
            df2[i] += -1

    df["zetels-huidig"] = df2
    return df

def seats(parties,df):
    
    count = 0
    for i in range(0,len(df)):
        if df["Naam"][i] in parties:
            count += df["zetels-huidig"][i]
    return count

def distance(parties,table):
    totaldistance = 0
    for i in range(0,len(parties)):
        for j in range(0,len(parties)):
            party1 = parties[i]
            party2 = parties[j] 
            totaldistance += table[party1][party2]**2
    return totaldistance / ((len(parties)**2 +len(parties))/2) * (1+(len(parties)/50))
          
def partyhate(parties):
    if "PVV" in parties:
        if "VVD" in parties:           
            parties = []
        elif "CDA" in parties:           
            parties = []
        elif "D66" in parties:           
            parties = []
        elif "VOLT" in parties:           
            parties = []
        elif "GL" in parties:           
            parties = []
    elif "FVD" in parties:
        if "VVD" in parties:           
            parties = []
        elif "CDA" in parties:           
            parties = []
        elif "D66" in parties:           
            parties = []
        elif "VOLT" in parties:           
            parties = []
        elif "GL" in parties:           
            parties = []
    return parties 


def possiblecombinations(allparties,table,df,P):
    totaldistance = 10000000
    seconddistance = 10000000
    thirddistance = 10000000
    regering = ""
    secondregering = ""
    thirdregering = ""
    stuff = allparties
    for L in range(len(stuff) + 1):
        for subset in itertools.combinations(stuff, L):
            
            mogregering = list(subset)
            if  np.random.normal(0,1) < 1.96:
                mogregering = partyhate(mogregering)
            if len(mogregering) > P:
                return [regering, secondregering, thirdregering,totaldistance,seconddistance,thirddistance]
            if seats(mogregering,df) > 75:
                if distance(mogregering,table) < totaldistance:
                    regering = mogregering
                    totaldistance = distance(mogregering,table)
                elif distance(mogregering,table) < seconddistance:
                    secondregering = mogregering
                    seconddistance = distance(mogregering,table)
                elif distance(mogregering,table) <thirddistance:
                    thirdregering = mogregering
                    thirddistance = distance(mogregering,table)
            
    return [regering, secondregering, thirdregering,totaldistance,seconddistance,thirddistance]




    
    
    
P = 6
sigma = 0.45
N = 40
np.zeros(N)
man = pd.DataFrame(np.zeros((N,6)),columns=["reger","reger2","reger3","dis1","dis2","dis3"] )
table = tablecreator(table,df)


for i in range(0,N):
    
    df = pd.read_excel("politiek.xlsx")
    df = df.merge(dataframe2, left_on='Naam',right_index=(True))
    df["Zetels peiling "] = df[df.columns[-1]]
    df = df.drop(columns=[df.columns[-1]])
    
    df3 = montecarloelection(df,sigma,dataframe)
    note = possiblecombinations(allparties,table,df3,P)
    man["reger"][i]= ", ".join(note[0])
    man["reger2"][i]= ", ".join(note[1])
    man["reger3"][i]= ", ".join(note[2])
   

ratio1 = man['reger'].value_counts()/N
ratio2 = man['reger2'].value_counts()/N
ratio3 = man['reger3'].value_counts()/N


ratio1 =  ratio1.reset_index()
ratio2 =  ratio2.reset_index()
ratio3 =  ratio3.reset_index()
full = pd.merge(ratio1,ratio2,how = "outer")
full = pd.merge(full,ratio3,how = "outer")

full["prefer"] = 200.0
for i in range(0,len(full)):
    string = full["index"][i]
    li = list(string.split(", "))
    full["prefer"][i] = distance(li,table)
full = full.fillna(0)

partygov = pd.DataFrame(np.zeros((4,len(allparties))),columns=allparties )
for j in allparties:
    for i in range(0,len(full)):
        string = full["index"][i]
        li = list(string.split(", "))
        if j in li:
            partygov[j][0] += 100 * full["reger"][i]
            partygov[j][1] += 100 * full["reger2"][i]
            partygov[j][2] += 100 * full["reger3"][i]
    
    partygov[j] = partygov[j]/100
    partygov[j][3] = (0.60 * partygov[j][0] + 0.30 * partygov[j][1] + 0.1 * partygov[j][2])
    

fig = plt.figure()
mylabels = ["Regering", "Oppositie"]
for j in range(0,len(allparties)):
    i = list(allparties)[j]
    plt.subplot(4, 5, j+1)
    plt.pie([partygov[i][3],1-partygov[i][3]])
    plt.title(i, y=1.0, pad=-1)
fig.legend( mylabels, loc='lower right')
    

full2 = full.drop(columns=['reger2', 'reger3'])





