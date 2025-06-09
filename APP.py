import streamlit as st
import pandas as pd
import numpy as np
import joblib

#load model and feature list    
model=joblib.load("xgb_model.pkl")
features=joblib.load("features.pkl")

st.title("USER VS BOT CLASSFICATION")

#inputs
input_data={}
for feature in features:
    input_data[feature] = st.number_input(f"{feature}, value=0")

#predection
if st.button("predict"):
    input_df=pd.DataFrame([input_data])
    prediction= model.predict(input_df)[0]
    proba= model.predict_proba(input_df)[0][1]


    st.write(f"**prediction:** {"Bot" if prediction == 1 else "User"}")
    st.write(f"**probability of being a Bot:** {proba:.2f}")


