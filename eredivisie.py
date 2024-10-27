# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 14:31:15 2022

@author: woute
"""

url2 = "https://www.vi.nl/competities/nederland/eredivisie/2022-2023/wedstrijden"   

import pandas as pd
import numpy as np

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
from concurrent import futures
driver = webdriver.Chrome()
from selenium.webdriver.common.by import By
from io import StringIO

driver = webdriver.Chrome()
url1 = "https://vi.nl"

driver.get(url1)
driver.find_element_by_xpath("//button[@class=' css-mnxivd']").click()
driver.find_element_by_xpath("//button[@class=' css-19ukivv']").click()
    

    
      

driver.get(url2)

continue_link = driver.find_element_by_tag_name('a')
elem = driver.find_elements_by_xpath("//*[@href]")

elems = driver.find_elements_by_tag_name('a')
dataframe = []
for elem in elems:
    href = elem.get_attribute('href')
    if href is not None:
        dataframe.append(href)

df= pd.DataFrame(dataframe)
df.rename(columns={ df.columns[0]: "URL" }, inplace = True)

df = df.URL

df= df[df.str.contains('https://www.vi.nl/wedstrijden/')]

df = df.reset_index()

df = df.URL

teams = df[0].split("sie/")
totalteams = teams[1].split("-vs-")
for i in range(1,9):
    teams = df[i].split("sie/")
    teams = teams[1].split("-vs-")
    totalteams = totalteams + teams 

table = pd.DataFrame()
table = pd.DataFrame(columns=totalteams,index=totalteams)
stand = pd.DataFrame(columns = ["Punten" , "Gespeeld" , "W", "G" , "V", "DS"], index=totalteams)
stand.Punten = 0 
stand.Gespeeld = 0 
stand.W = 0 
stand.G = 0 
stand.V = 0 
stand.DS = 0 
table.loc[totalteams[0]][totalteams[1]]

        





checknoplayoffs = []
for i in range(0,len(df)):
    url1 = df[i]

    driver.get(url1)
    
    txt = driver.find_element_by_xpath("/html/body").text
    txt = txt.split("Schoten op doel")
    txt = txt[1].split("Balbezit")
    txt = txt[0]
    txt = "Doelschoten" + txt
    df2 = pd.read_csv(StringIO(txt))
    teams = df[i].split("sie/")
    teams = teams[1].split("-vs-")
    matchup = teams[0] + teams[1]
    if matchup in checknoplayoffs:
        print("repeat detected" + url1)
    else:       
        checknoplayoffs.append(matchup)
        table.loc[teams[0]][teams[1]] = df2.Doelschoten[0] - df2.Doelschoten[1]
    
driver.quit()
       
for i in totalteams:
    for j in totalteams:
        value = table.loc[i][j]
        if value > 0:
            stand.Punten[i] = stand.Punten[i] + 3
            stand.W[i] = stand.W[i] + 1
            stand.V[j] = stand.V[j] + 1
            stand.Gespeeld[i] =  stand.Gespeeld[i]  + 1 
            stand.Gespeeld[j] =  stand.Gespeeld[j]  + 1 
            stand.DS[i] =  stand.DS[i]  + value
            stand.DS[j] =  stand.DS[j]  - value
        elif value == 0:
            stand.Punten[i] = stand.Punten[i] + 1
            stand.Punten[j] = stand.Punten[j] + 1
            stand.G[i] = stand.G[i] + 1
            stand.G[j] = stand.G[j] + 1
            stand.Gespeeld[i] =  stand.Gespeeld[i]  + 1 
            stand.Gespeeld[j] =  stand.Gespeeld[j]  + 1 
            stand.DS[i] =  stand.DS[i]  + value
            stand.DS[j] =  stand.DS[j]  - value
        elif value < 0:
            stand.Punten[j] = stand.Punten[j] + 3
            stand.W[j] = stand.W[j] + 1
            stand.V[i] = stand.V[i] + 1
            stand.Gespeeld[i] =  stand.Gespeeld[i]  + 1 
            stand.Gespeeld[j] =  stand.Gespeeld[j]  + 1 
            stand.DS[i] =  stand.DS[i]  + value
            stand.DS[j] =  stand.DS[j]  - value
        else: 
            hoi = 2
        
stand = stand.sort_values(by = ["Punten"], ascending = False )