# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("bballfinal3.csv")

dates = sorted(set(df["date"]))

players = sorted(set(df["uniquedone"]))

teams = sorted(set(df["Tmr"]))

playersteams = players + teams 

df2 = pd.DataFrame(index = dates,columns = players)

M = len(df)
N = len(dates)
K = len(players)

#for i in range(0,M):
#    df2[df["uniquedone"][i]][df["date"][i]] = df["BPM"][i]

start = 0 
bpmgas = pd.DataFrame(index = dates,columns = players)


for i in range(0,K):
    bpmgas[players[i]][dates[0]] = start
    


df5 = pd.DataFrame()

df["indicator"] = df["date"] + df["Tmr"]
df["dateopp"] = df["date"] + df["Opp"]

df5["date"] = df["date"]
df5["Tmr"] = df["Tmr"]
df5["Opp"] =  df["Opp"] 
df5["dateopp"]  = df["dateopp"]
df5["indicator"] = df["indicator"]
df5["home"] = ""
df5["away"] = ""
df5 = df5.drop_duplicates()
df5 = df5.reset_index()
G = len(df5)
#gameschedule 
df5["homeBPM"]=0
df5["awayBPM"] = 0 
df5["homepoints"]=0
df5["awaypoints"] = 0 
df5["Tmrpoints"] = 0
df5["Opppoints"] = 0
df5["Win"] = 0
df5 = df5.set_index("indicator")


for i in range(0,M):
    df5["Tmrpoints"][df["indicator"][i]] = df5["Tmrpoints"][df["indicator"][i]] + df["PTS"][i]
    if i % 10000 == 0: 
        print(i)
df5 = df5.reset_index()


list1 = set(df5["indicator"]) - set(df5["dateopp"]) 
list2 = set(df5["dateopp"]) - set(df5["indicator"])
list3 = set.union(list1, list2)
for j in list3:
    df5 = df5[df5.indicator != j]
    df5 = df5[df5.dateopp != j]



for i in range(0,G):
    list = sorted([df5["Tmr"][i],df5["Opp"][i]])
    df5["home"][i] = list[0]
    df5["away"][i] = list[1]
    if df5["home"][i] == df5["Tmr"][i]:
        df5["homepoints"][i] =  df5["Tmrpoints"][i]
    else:
        df5["awaypoints"][i] =  df5["Tmrpoints"][i]
        if i % 10000 == 0: 
            print(i)    
       
        
df5 = df5.sort_values(['date', 'home'], ascending=[True, True])
del df5['level_0']
df5 = df5.reset_index()


for i in range(0,G-1):
    if df5["home"][i] == df5["home"][i+1]:
        if df5["date"][i] == df5["date"][i+1]:

            if df5["homepoints"][i] == 0:
                df5["homepoints"][i] = df5["homepoints"][i+1]
                df5["awaypoints"][i+1] = df5["awaypoints"][i]
                if i % 10000 == 0: 
                    print(i)
            else:
                df5["awaypoints"][i] = df5["awaypoints"][i+1]
                df5["homepoints"][i+1] = df5["homepoints"][i]
    
df6 = pd.DataFrame()
df6["date"] = df5["date"]
df6["home"] = df5["home"]
df6["away"] = df5["away"]
df6["homepoints"] = df5["homepoints"]
df6["awaypoints"] = df5["awaypoints"]

df6 = df6.drop_duplicates()                  
df6 = df6.reset_index()
schedule = schedule.sort_values(by=['date'])



for i in range(0,10000)
    df["Tmr"][i]
        
    if df["Tmr"][i] == schedule["home"]:
        schedule["homepoints"][df["date"][i]] = schedule["homepoints"][df["date"][i]] + df["PTS"][i]
    else:
        schedule["awaypoints"][df["date"][i]] = schedule["awaypoints"][df["date"][i]] + df["PTS"][i]


    