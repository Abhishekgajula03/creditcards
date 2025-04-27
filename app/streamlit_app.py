import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from streamlit_extras.let_it_rain import rain

# Function to set background image
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('data:image/jpeg;base64,{b64_string}');
            background-size: cover;
            background-position: center;
        }}
        .hover-btn:hover {{
            color: white;
            background-color: green;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(page_title="Credit Risk Predictor", page_icon=":money_with_wings:", layout="centered")

# Apply background
set_background("assets/background.jpg")

st.title("💳 Loan Applicant Credit Risk Prediction")

# Load model artifacts
artifacts = joblib.load("../models/credit_model.pkl")
model = artifacts['model']
encoder = artifacts['encoder']
scaler = artifacts['scaler']

st.header("📝 Fill Applicant Details")

with st.form("credit_form"):
    sex = st.selectbox('Sex', ['male', 'female'])
    housing = st.selectbox('Housing', ['own', 'free', 'rent'])
    saving = st.selectbox('Saving accounts', ['little', 'moderate', 'quite rich', 'rich', 'Unknown'])
    checking = st.selectbox('Checking account', ['little', 'moderate', 'rich', 'Unknown'])
    purpose = st.selectbox('Purpose', ['radio/TV', 'education', 'furniture/equipment', 'car', 'business'])
    age = st.slider('Age', 18, 75, 30)
    job = st.selectbox('Job', [0, 1, 2, 3])
    amount = st.number_input('Credit amount', 250, 20000, 1000)
    duration = st.slider('Duration (months)', 4, 72, 12)

    submit = st.form_submit_button("💸 Predict", help="Click to predict credit risk.")

if submit:
    input_df = pd.DataFrame([[age, job, amount, duration, sex, housing, saving, checking, purpose]],
                            columns=['Age', 'Job', 'Credit amount', 'Duration', 'Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose'])

    X_cat = encoder.transform(input_df[['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']])
    X_num = scaler.transform(input_df[['Age', 'Job', 'Credit amount', 'Duration']])
    X = np.hstack((X_num, X_cat))

    prediction = model.predict(X)[0]
    result = "✅ Good Credit Risk" if prediction == 0 else "⚠️ Bad Credit Risk"

    st.success(result)

    # Rain effect based on prediction
    emoji = "💸" if prediction == 0 else "⚠️"
    rain(emoji=emoji, font_size=54, falling_speed=5, animation_length="infinite")