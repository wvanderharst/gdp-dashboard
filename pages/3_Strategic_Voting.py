import streamlit as st


st.title("Stratagy")
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
