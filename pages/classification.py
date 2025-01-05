import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import google.generativeai as genai
import time
import os

HEART_DISEASE_WARNING_THRESHOLD = 0.50


def generate_response(question: str):
    api_key = st.secrets["GENAI_APIKEY"]
    model_selected = st.secrets["GENAI_MODEL"]

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_selected)

    response = model.generate_content(question)
    return response.text


def load_model():
    return tf.keras.models.load_model("./models/default_model.h5")


# read dataset
data_encode_columns = [
    'Smoking',
    'Stroke',
    'PhysicalHealth',
    'DiffWalking',
    'AgeCategory',
    'Diabetic',
    'PhysicalActivity',
    'KidneyDisease',
    'HeartDisease'
]

st.title("About Heart Disease")
st.text("The Heart Disease Prediction System is a machine learning-based tool designed to assess an individual's risk of developing heart disease using key health and lifestyle parameters. It analyzes inputs such as smoking habits, history of stroke, physical health status, difficulty walking, age category, diabetes status, physical activity levels, and history of kidney disease. By processing these factors, the system predicts the likelihood of heart disease and categorizes the risk as low, moderate, or high. This allows users to gain insights into their health and take preventive measures, such as lifestyle adjustments or consulting a healthcare professional, for early intervention. The system aims to support proactive health management and improve overall well-being.")

# Smoking input
st.markdown("---")
st.text("1. Please select 'Yes' if you are currently smoking, otherwise select 'No'. Smoking is a major risk factor for heart disease.")

smoking = st.selectbox(
    label="smoking",
    options=['Yes', 'No']
)

# Stroke input
st.markdown("---")
st.text("2. Have you ever experienced a stroke? Select 'Yes' if you have, or 'No' if you haven't. Stroke history can increase the risk of heart disease.")

stroke = st.selectbox(
    label="stroke",
    options=['Yes', 'No']
)

# PhysicalHealth input
st.markdown("---")
st.text("3. How many days in the past 30 days did you feel that your physical health was not good? (0-30 days)")

PhysicalHealth = st.selectbox(
    label="physical health",
    options=[int(option) for option in sorted([
        3., 0., 20., 28., 6., 15., 5., 30., 7., 1., 2., 21., 4.,
        10., 14., 18., 8., 25., 16., 29., 27., 17., 24., 12., 23., 26.,
        22., 19., 9., 13., 11.
    ])]
)

# DiffWalking input
st.markdown("---")
st.text("4. Do you have difficulty walking or moving? Select 'Yes' if you have difficulty, or 'No' if you don't.")

diffwalking = st.selectbox(
    label="diff walking",
    options=['Yes', 'No']
)

# AgeCategory input
st.markdown("---")
st.text("5. Please select your age range.")

AgeCategory = st.selectbox(
    label="AgeCategory",
    options=[option for option in [
        '18-24', '25-29', '30-34', '35-39',
        '40-44', '45-49', '50-54', '55-59', '60-64',
        '65-69', '70-74', '75-79', '80 or older'
    ]]
)

# Diabetic input
st.markdown("---")
st.text("6. Do you have diabetes? Select the appropriate option that applies to you.")

diabetic = st.selectbox(
    label="Diabetic",
    options=['Yes', 'No', 'No, borderline diabetes', 'Yes (during pregnancy)']
)

# PhysicalActivity input
st.markdown("---")
st.text("7. Do you engage in physical activity regularly? Select 'Yes' if you do, or 'No' if you don't.")

physical_activity = st.selectbox(
    label="PhysicalActivity",
    options=['Yes', 'No']
)

# KidneyDisease input
st.markdown("---")
st.text("8. Do you have a history of kidney disease? Select 'Yes' if you have, or 'No' if you don't.")

KidneyDisease = st.selectbox(
    label="KidneyDisease",
    options=['Yes', 'No']
)


if st.button("Submit", type="primary"):
    all_values = [
        smoking,
        stroke,
        PhysicalHealth,
        diffwalking,
        AgeCategory,
        diabetic,
        physical_activity,
        KidneyDisease
    ]

    columns = data_encode_columns[:-1]

    for idx, (col, value) in enumerate(zip(columns, all_values)):
        for dir in os.listdir("./models/"):
            if "pkl" in dir and col in dir:
                if os.path.getsize(f"./models/label_cndoding_{col}.pkl") > 0:
                    with open(f"./models/label_cndoding_{col}.pkl", 'rb') as file:
                        encoder = pickle.load(file)
                        all_values[idx] = encoder.transform(
                            [all_values[idx]])[0]
                else:
                    st.text("Encoder are not there!")

    model = load_model()
    pred = np.squeeze(model.predict(np.expand_dims(all_values, axis=0)))

    pred_percentage = pred * 100
    pred = round(pred_percentage, 2)

    if pred > HEART_DISEASE_WARNING_THRESHOLD:
        st.warning(
            f"Your probability of having heart disease symptoms is high. ({pred}%)")

    else:
        st.success(
            f"Your probability of not having heart disease symptoms is high. ({pred}%)")

    with st.spinner("Generating suggestion..."):
        placeholder = st.empty()
        typed_text = ""

        question = f"Imagine there is a patient with a probability of getting heart diseases {pred}%. What is your recommendation to that patient?"
        answer = generate_response(question)

        if isinstance(answer, str):
            for char in answer:
                typed_text += char
                placeholder.markdown(typed_text)
                time.sleep(0.01)
        else:
            st.error("Error: Invalid response format.")
