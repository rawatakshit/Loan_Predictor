import streamlit as st
import pickle
import pandas as pd
from PIL import Image

model= pickle.load(open('model.sav','rb'))

st.title("Loan Prediction ")
st.sidebar.header("Applicant Information")
image=Image.open('loan.webp')
st.image(image)

def info():
    no_of_dependents = st.sidebar.slider("Number of Dependents",0,10,1)
    income_annum = st.sidebar.slider("Annual Income",0,100000000,1)
    loan_amount = st.sidebar.slider("Loan Amount ",0,100000000,1)
    loan_term = st.sidebar.slider("Loan Term",0,30,1)
    cibil_score = st.sidebar.slider("Cibil Score",0,900,1)
    residential_assets_value = st.sidebar.slider("Residential Assets Value",0,100000000,1)
    commercial_assets_value = st.sidebar.slider("Commercial Assets Value",0,100000000,1)
    luxury_assets_value = st.sidebar.slider("Lusury Assets Value",0,100000000,1)
    bank_asset_value = st.sidebar.slider("Bank Assets Value",0,100000000,1)

    user_datafn={
        " no_of_dependents" : no_of_dependents,
        " income_annum" : income_annum,
        ' loan_amount' :  loan_amount,
        ' loan_term' : loan_term,
        " cibil_score" : cibil_score,
        " residential_assets_value" : residential_assets_value,
        " commercial_assets_value" : commercial_assets_value,
        " luxury_assets_value" : luxury_assets_value,
        " bank_asset_value" : bank_asset_value,
    }
    final = pd.DataFrame(user_datafn,index=[0])
    return final

user_data=info()
st.header("Applicant information")
st.write(user_data)

approval=model.predict(user_data)
st.subheader("Status")

# st.subheader(approval) was showing percentage. Therefore if condition
if approval>1:
    st.subheader("Approved")
else:
    st.header("Rejected")
