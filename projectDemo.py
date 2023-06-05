import streamlit as st
import pandas as pd
import numpy as np
from DeploymentTools.classify import predict, scaleFeatures

st.title('Credit Risk Assessment')
st.markdown('This is a project demo for model deployment VIX rakamin x ID/X Partner by Muhammad Khalifa Umana')

st.header("Necessary feature")
col1, col2 = st.columns(2)

with col1:
    st.text("Monetary information")
    loan_amnt = st.number_input('Loan Amount', min_value=500)
    annual_inc = st.number_input('Annual Income')

with col2:
    st.text("User information")
    open_acc = st.number_input('Account opened')
    dti = st.number_input('dti (%)', max_value=40)

st.text("Inquiry information")
inq_last_6mths = st.number_input('amount of inquiries for last 6 months')


st.text('')
if st.button("Predict assesment"):
    scaledFeat = scaleFeatures(loan_amnt, annual_inc)
    feat = np.concatenate((scaledFeat, [dti, open_acc, inq_last_6mths]))
    result, proba = predict(
        np.array([feat]))
    if result == 0:
        st.text('Bad Loan')
        st.text('probability : '+ str(round(proba[0][0]*100,2))+'%')
    else:
        st.text('Good Loan')
        st.text('probability : '+ str(round(proba[0][0]*100,2))+'%')