# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 09:55:16 2021

@author: woute
"""

import pandas as pd
import numpy as np
import math as math 
import scipy.special as sc
from random import randint
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import matplotlib.pyplot as plt
import statistics as st 

def schedulesync(schedule,df):
    test = len(df)
    word = df["date"][test-1]
    letter = 0
    remember = len(df)
    for i in range(0,len(schedule)):
        if letter == 1:
            if word != schedule["date"][i]:
                remember = i
                letter = 0 
        elif word == schedule["date"][i]:
            letter = 1 
    schedule = schedule[0:remember-1]
    return schedule
    


def RunningGAS(df,playerewma,teamsewmatrue,teams,limit,theta):
    datus = ""
    number = -1
    y = 0
    N = len(df["date"])
    for i in range(0,N):
        if i % 10000 == 0: 
            if i < 10000:
                teams = sorted(set(df["Tmr"][0:10000]))
            elif i+10000 > N:
                teams = sorted(set(df["Tmr"][N-10000:N]))      
            else:
                teams = sorted(set(df["Tmr"][i-10000:i+10000]))
        if datus != df["date"][i]:
            #print(i)
            datus = df["date"][40]
            number  = number + 1
            teamsin = set(df["Tmr"][(df.date == datus)])
            for j in teamsin:
                frut = df["uniquedone"][(df.date == datus) & (df.Tmr == j)]
                scores = playerewma[frut].iloc[number]
                lengthscores = len(scores)
            
                if lengthscores > 10:
                   scores = sorted(scores)[3:]
                
                elif lengthscores > 9:
                    scores  = sorted(scores)[2:]
                elif lengthscores > 8:
                   scores =  sorted(scores)[1:]
                else:
                    scores = sorted(scores)
                playerewma[j][number:] = st.mean(scores)


            teamsewmatrue[teams][number:number+1] = playerewma[teams][number:number+1] - playerewma[teams][number:number+1].mean(axis =1)[0]
            for j in teams :
                teamsewmatrue[j][number] = playerewma[j][number] - playerewma[teams][number:number+1].mean(axis =1)[0]            
        #if playerewma[df["uniquedone"][i]][number] == 0.0:
        #    playerewma[df["uniquedone"][i]][number] = (df["BPM"][i])/10
        
        fsubset= np.array([playerewma[df["uniquedone"][i]][number],teamsewmatrue[df["Opp"][i]][number]])
        x = [df["BPM"][i],df["homecourt"][i],theta[2]]
        if  x[0] > limit:
            if df["MPr"][i] < 8:
                x[0] = limit
        elif x[0] <= limit:
            if df["MPr"][i] < 8: 
                x[0] = -limit
        playerewma[df["uniquedone"][i]][number+1:] = GAS2(fsubset,x,theta[0],theta[1])
    
def nabla(x,y,fsubset):
    labda1 = np.exp(fsubset[0]-fsubset[1]/5)
    labda2 = np.exp(0)
    nab = np.array([x-labda1,y-labda2])
    return nab

def nabla2(z,fsubset):
    if z[1] == 1:
        labda1 = np.exp(fsubset[0] + z[2] -fsubset[1])
        labda2 = np.exp(0)
    else:
        labda1 = np.exp(fsubset[0]-fsubset[1]- z[2])
        labda2 = np.exp(0)
    root = 2*(labda1 * labda2)**0.5
    zabs = abs(z[0])
    w = root * sc.iv(zabs+1,root) / sc.iv(zabs,root)
    nab = np.array([z[0]-labda1+w, labda2 + w ])
    return nab



def GAS(fsubset,x,y,alfa1,beta1):
    omega = 0 
    attra = np.array([alfa1,alfa1])  * nabla(x,y,fsubset) 
    bttra = np.array([beta1,beta1])  * fsubset    
    gast = omega + attra + bttra
    return gast[0]

def GAS2(fsubset,z,alfa1,beta1):
    omega = 0 
    attra = np.array([alfa1,alfa1])  * nabla2(z,fsubset) 
    bttra = np.array([beta1,beta1])  * fsubset    
    gast = omega + attra + bttra
    return gast[0]

def DoublePoison(x,y,fsubset):
    labda1 = np.exp(fsubset[0]-fsubset[1]/5)
    labda2 = np.exp(0)
    bigp = np.exp(-labda1-labda2) * (labda1 ** x / math.gamma(x+1)) *(labda2 ** y / math.gamma(y+1))
    return bigp


def winninghanceactual(fsubset):
    reverse_fsubset = np.array([0,0])
    reverse_fsubset[0] = fsubset[1]
    reverse_fsubset[1] = fsubset[0]
    odds1 = winningchance(fsubset)
    odds2 = winningchance(reverse_fsubset)
    odds3 = np.array([(odds1[0]+odds2[1])/2, (odds1[1]+odds2[0])/2] )
    return odds3 

def winningchance(fsubset):
    wop = 0 
    for i in range(-100,0):
        wop = wop + Skellam([i,0,0],fsubset)
    wop2 = Skellam([0,0,0],fsubset)
    wop3 = 0
    for i in range(1,100):
        wop3 = wop3 + Skellam([i,0,0],fsubset)
    woptotal = wop + wop3
    wopshare = wop / woptotal
    wop3share = wop3 / woptotal
    
    wop = wop + wop2 * wopshare
    wop3 = wop3 + wop2 * wop3share
    winandloss = np.array([wop,wop3])
    return winandloss

def schedulewithf(teamsewmatrue,schedule):
    N = len(schedule["date"])
    schedule["fhome"] = 0.0 
    schedule["faway"] = 0.0 
    for i in range(0,N):
        schedule["fhome"][i] = teamsewmatrue[schedule["home"][i]][schedule["date"][i]]
        schedule["faway"][i] = teamsewmatrue[schedule["away"][i]][schedule["date"][i]]

def Skellam(z,fsubset):
    if z[1] == 1:
        labda1 = np.exp(fsubset[0] + z[2])
        labda2 = np.exp(fsubset[1])
    else:
        labda1 = np.exp(fsubset[0])
        labda2 = np.exp(fsubset[1]+z[2])
        
    root = 2* (labda1 * labda2)**0.5
    bigp = np.exp(-labda1-labda2) * (labda1/labda2) ** (z[0]/2) * sc.iv(z[0],root)
    return bigp
        


def skellamlog(schedule,ini_period,theta):
    N = np.size(schedule,0)
    loglik = 0.0
    #loglik2 = 0.0
    for q in range(ini_period,N):
            fsubset = np.array([schedule["fhome"][q],schedule["faway"][q]])
            #fsubset2 = np.array([schedule["faway"][q],schedule["fhome"][q]])
            adjustedscore = schedule["net"][q]
            #if adjustedscore > 6:
            #    adjustedscore = 6
            #elif adjustedscore < -6:
            #    adjustedscore = -6
                
            z = [adjustedscore,1,theta[3]]
            
            #z2 = [adjustedscore,0,theta[3]-0.5]
            loglik = loglik + np.log(Skellam(z,fsubset))
            #loglik2 = loglik2 + np.log(Skellam(z2,fsubset2))
            
            #print(z[0]," ", np.log(Skellam(z,fsubset))," ",np.log(Skellam(z2,fsubset)))
    return loglik


def plotwinners(teamsewmatrue,team1,team2):
    teamsewmatrue["odds"] = 0.5
    teamsewmatrue["rollodds"] = 0.5
    N = len(teamsewmatrue)
    for i in range(0,N):
        fsubset = [teamsewmatrue[team1][i], teamsewmatrue[team2][i]]
        teamsewmatrue["odds"][i]= winningchance(fsubset)[1]
    for i in range(19,N):
        teamsewmatrue["rollodds"][i] =  teamsewmatrue["odds"][i-20:i].mean()
    teamsewmatrue["rollodds"][0:N].plot()


def guesswinnercorrect(schedule):
    N = len(schedule)
    win = 0 
    win2 = 0
    count2 = 0
    win3 = 0
    count3 = 0 
    win4 = 0
    count4 = 0 
    win5= 0
    count5 = 0         
    for i in range(0,N):
        if schedule["fhome"][i] >= schedule["faway"][i]:
            if schedule["win"][i] == 1:
                win = win + 1
        else:
            if schedule["win"][i] == 0:
                win = win + 1
        
        if schedule["fhome"][i] >= schedule["faway"][i]+0.5:
            count2 = count2 + 1
            if schedule["win"][i] == 1:
                win2 = win2 + 1
        elif schedule["faway"][i] > schedule["fhome"][i]+0.5:
            count2 = count2 + 1
            if schedule["win"][i] == 0:
                win2 = win2 + 1
        if schedule["fhome"][i] >= schedule["faway"][i]+1:
            count3= count3 + 1
            if schedule["win"][i] == 1:
                win3 = win3 + 1
        elif schedule["faway"][i] > schedule["fhome"][i]+1:
            count3 = count3 + 1
            if schedule["win"][i] == 0:
                win3 = win3 + 1

   
        if schedule["fhome"][i] >= schedule["faway"][i]:
            if schedule["fhome"][i] <= schedule["faway"][i]+0.5:
                count4= count4 + 1
                if schedule["win"][i] == 1:
                    win4 = win4 + 1
            
        elif schedule["faway"][i] >= schedule["fhome"][i]:
            if schedule["faway"][i] <= schedule["fhome"][i]+0.5:
                count4 = count4 + 1
                if schedule["win"][i] == 0:
                    win4 = win4 + 1
        
        if schedule["fhome"][i] >= schedule["faway"][i]:
            if schedule["fhome"][i] <= schedule["faway"][i]+0.1:
                count5= count5 + 1
                if schedule["win"][i] == 1:
                    win5 = win5 + 1
            
        elif schedule["faway"][i] >= schedule["fhome"][i]:
            if schedule["faway"][i] <= schedule["fhome"][i]+0.1:
                count5 = count5 + 1
                if schedule["win"][i] == 0:
                    win5 = win5 + 1                
    return([win/N])      
    #return([win/N,win2/count2,win3/count3,win4/count4,win5/count5])
        

def main(theta,df,playerewma,teamsewmatrue,teams,schedule,limit,ini_period):    
    #initialize
    RunningGAS(df, playerewma, teamsewmatrue, teams,limit,theta)
    
    schedulewithf(teamsewmatrue,schedule)
     
    loglik = -skellamlog(schedule,ini_period,[0,0,0,0.33])
    print(theta[0]," ",theta[1]," ",theta[2]," " ,loglik)
    return loglik



df = pd.read_csv("bballfinal4.csv")
df =df.sort_values(by=['date'])
del df['level_0']
df = df.reset_index()
    
schedule = pd.read_csv("schedule2.csv")
schedule["net"] = schedule["homepoints"] - schedule["awaypoints"]
schedule["win"] = 0
for i in range(0,len(schedule)):
    if  schedule["net"][i] > 0:   
        schedule["win"][i] = 1
    
df = df[['Tmr', 'Opp',"MPr","BPM","date","uniquedone","homecourt","PTS"]]
#test = 20000
#df = df[0:test]
schedule = schedulesync(schedule,df)
dates = sorted(set(df["date"]))

players = sorted(set(df["uniquedone"]))


playerewma = pd.DataFrame(index = dates,columns = players)
playerewma.loc[:,:] = 0.0

teams = sorted(set(df["Tmr"]))


playerewma[teams]= 0.0

teamsewmatrue = playerewma[teams]

#bnds = ((0.00000000001, 0.1), (0.9, 0.99999))

theta = [0.015 , 0.98 ,0.1]

ini_period = 10
limit = 25

linear_constraint = LinearConstraint([[1, 1,0],[1,0,0],[0,1,0],[0,0,1]],[-1,-0.3,0.9,0],[1,0.3,0.999999,1])



theta = [0.005107164345546121,   0.9948963917593507,   0.03817877555472801]
main(theta,df,playerewma,teamsewmatrue,teams,schedule,limit,ini_period)
tSol = minimize(main, theta,args=(df,playerewma,teamsewmatrue,teams,schedule,limit,ini_period), constraints = linear_constraint,  
                options={'disp': True, 'maxiter':15})




theta_new = tSol.x
import matplotlib.pyplot as plt
df17 = teamsewmatrue[["LAL","CHI"]]
df18 = df17.rolling(window=20).mean()
df18.plot()
plt.show()

df17 = playerewma[["Michael JordanG-F0","Magic JohnsonG-F0","LeBron JamesF-G0","Vince CarterG-F0"]]
df18 = df17.rolling(window=20).mean()
df18.plot()
plt.show()

#main(theta,df,playerewma,schedule,ini_period) constraints=linear_constraint,

main(theta,df,playerewma,teamsewmatrue,teams,schedule,limit,ini_period)

[df["homecourt"] == 1 ]
means = df.groupby('homecourt').mean()
means = schedule.groupby('win').mean()
means = schedule.mean()



guesswinnercorrect(schedule)

playerewma.to_csv("initialplayer")
teamsewmatrue.to_csv("initialteam")




