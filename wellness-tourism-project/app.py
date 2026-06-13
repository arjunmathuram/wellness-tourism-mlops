
import streamlit as st

st.title("Wellness Tourism Package Predictor")

st.success("Model loaded successfully.")

st.header("Customer Information")

name = st.text_input("Customer Name")

st.write("Name entered:", name)
