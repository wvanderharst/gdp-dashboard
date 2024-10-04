import altair as alt
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import ast
import itertools



def all_in_list(short_list, long_list):
    return set(short_list).issubset(set(long_list))

def convert_string_to_list(s):
    return ast.literal_eval(s)

def contains_all(long_list, short_list):
    return all(item in long_list for item in short_list)

def load_data2():
    df = pd.read_csv("https://raw.githubusercontent.com/wvanderharst/peilingen-dataset/refs/heads/main/data/partygov.csv")
    return df

def load_data3():
    df = pd.read_csv("https://raw.githubusercontent.com/wvanderharst/peilingen-dataset/refs/heads/main/data/elections.csv")
    return df

length = load_data3()
lengthN = len(length.columns)-2

df3 = load_data2()

# Show a multiselect widget with the genres using `st.multiselect`.
partij = st.selectbox(
    "Partij",
    df3.Partij.unique())
partij1 = partij.replace('/', '')

df = pd.read_csv(f'https://raw.githubusercontent.com/wvanderharst/peilingen-dataset/refs/heads/main/data_conditional/{partij1}.csv')

allparties = set(df3["Partij"])


df['reger'] = df['reger'].apply(convert_string_to_list)


df34= df[df['reger'].apply(lambda x: partij in x)]

st.dataframe(
    df['reger'].value_counts()/len(df['reger']))

df8 = df['reger'].value_counts().sum()
df34 = df34['reger'].value_counts().sum()
#df8 = (df['reger'].value_counts()/len(df['reger'])).sum()
#df34 = (df34['reger'].value_counts()/len(df34['reger'])).sum()
num = df34/df8
labels = 'Coaltitie', 'Oppositie'
sizes = [num, 1-num ]
explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()


ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
    shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title(partij)
st.pyplot(fig1)

