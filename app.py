import streamlit as st 
import pandas as pd

st.title("Hello world!")
st.write("This is the start of the CubeSat Diagnostic app.")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    st.write("File uploaded successfully!")
    df = pd.read_csv(uploaded_file)
    st.write("Here's a preview of the data:")
    st.dataframe(df.head())
else:    
  st.write("Please upload a file to proceed.")