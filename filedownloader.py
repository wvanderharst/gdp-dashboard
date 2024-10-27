# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:23:19 2023

@author: woute
"""

import pandas as pd
import numpy as np



from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from datetime import datetime

def recentdate():
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

    data = pd.read_excel(folder + "\Results_Longitudinal.xlsx")
    return data 