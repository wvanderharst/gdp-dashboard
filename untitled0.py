# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:58:04 2022

@author: woute
"""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
from concurrent import futures
url1 = "https://vi.nl"
url = "https://stathead.com/basketball/pgl_finder.cgi?request=1&lg_id=NBA&order_by=pts&match=game&season_start=1&year_max=2022&is_playoffs=Y&season_end=-1&order_by_asc=0&age_min=0&age_max=99&year_min=2021&offset=0"

driver = webdriver.Chrome()
driver.get(url1)

login = driver.find_element_by_xpath("//input").send_keys("Elwoutertje")
password = driver.find_element_by_xpath("//input[@type='password']").send_keys("8m!rXZEQ!!Lvuyt")
driver.find_element_by_xpath("//button[@class=' css-47sehv']").click()
submit = driver.find_element_by_xpath("//input[@value='Login']").click()

driver.get(url)
driver.page_source
soup = BeautifulSoup(driver.page_source, 'html.parser')
table = soup.find(id='stats')
df = pd.read_html(str(table))[0]

sting = list(range(2000))

for i in range(0,2000):
    j = str(100 * i)
    sting[i] = "https://stathead.com/basketball/pgl_finder.cgi?request=1&lg_id=NBA&order_by=pts&match=game&season_start=1&year_max=2022&is_playoffs=Y&season_end=-1&order_by_asc=0&age_min=0&age_max=99&year_min=2021&offset=" + j


for i in sting:
    driver.get(i)
    print(i)
    driver.page_source
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    table = soup.find(id='stats')
    df2 = pd.read_html(str(table))[0]
    df = df.append(df2)


df.to_csv("dataframemissing3.csv")
#def selenium_work(url):

#    soup = BeautifulSoup(driver.page_source, 'html.parser')
#    table = soup.find(id='stats')
#    df = pd.read_html(str(table))[0]

#selenium_work(url)


#driver.quit()
