# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 18:33:35 2022

@author: woute
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 14:31:15 2022

@author: woute
"""

import pandas as pd
import numpy as np

df = pd.read_csv("linkseng.csv")
df = df[(df.Link == "Bekijk wedstrijd") ]

df = df.reset_index()

df = df.URL

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
from concurrent import futures
from selenium.webdriver.common.by import By
from io import StringIO

teams = df[0].split("gue/")
totalteams = teams[1].split("-vs-")
for i in range(1,10):
    teams = df[i].split("gue/")
    teams = teams[1].split("-vs-")
    totalteams = totalteams + teams 

table = pd.DataFrame()
table = pd.DataFrame(columns=totalteams,index=totalteams)
stand = pd.DataFrame(columns = ["Punten"], index=totalteams)
stand.Punten = 0 


        
            
            

driver = webdriver.Chrome()
url1 = "https://vi.nl"

driver.get(url1)
driver.find_element_by_xpath("//button[@class=' css-mnxivd']").click()
driver.find_element_by_xpath("//button[@class=' css-19ukivv']").click()
    

for i in range(0,380):
    url1 = "https://vi.nl" + df[i]

    driver.get(url1)
    
    txt = driver.find_element_by_xpath("/html/body").text
    txt = txt.split("Schoten op doel")
    txt = txt[1].split("Balbezit")
    txt = txt[0]
    txt = "Doelschoten" + txt
    df2 = pd.read_csv(StringIO(txt))
    teams = df[i].split("gue/")
    teams = teams[1].split("-vs-")
    table.loc[teams[0]][teams[1]] = df2.Doelschoten[0] - df2.Doelschoten[1]
    
    
for i in totalteams:
    for j in totalteams:
        value = table.loc[i][j]
        if value > 0:
            stand.Punten[i] = stand.Punten[i] + 3
        elif value == 0:
            stand.Punten[i] = stand.Punten[i] + 1
            stand.Punten[j] = stand.Punten[j] + 1
        elif value < 0:
            stand.Punten[j] = stand.Punten[j] + 3
        else: 
            hoi = 2
stand = stand.sort_values(by = ["Punten"], ascending = False )