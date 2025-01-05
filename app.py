import streamlit as st
import tensorflow as tf


def introduction():
    st.title("Heart Disease Prediction")
    st.subheader("Created by Kelompok 4")
    st.title("About Hearth Disease")
    st.text("According to the CDC, heart disease is a leading cause of death for people of most races in the U.S. (African Americans, American Indians and Alaska Natives, and whites). About half of all Americans (47%) have at least 1 of 3 major risk factors for heart disease: high blood pressure, high cholesterol, and smoking. Other key indicators include diabetes status, obesity (high BMI), not getting enough physical activity, or drinking too much alcohol. Identifying and preventing the factors that have the greatest impact on heart disease is very important in healthcare. In turn, developments in computing allow the application of machine learning methods to detect patterns in the data that can predict a patient's condition.")
    st.title("About Dataset")
    st.text("The dataset originally comes from the CDC and is a major part of the Behavioral Risk Factor Surveillance System (BRFSS), which conducts annual telephone surveys to collect data on the health status of U.S. residents. As described by the CDC: Established in 1984 with 15 states, BRFSS now collects data in all 50 states, the District of Columbia, and three U.S. territories. BRFSS completes more than 400,000 adult interviews each year, making it the largest continuously conducted health survey system in the world. The most recent dataset includes data from 2023. In this dataset, I noticed many factors (questions) that directly or indirectly influence heart disease, so I decided to select the most relevant variables from it. I also decided to share with you two versions of the most recent dataset: with NaNs and without it.")


classification_page = st.Page("pages/classification.py")
chatbot_page = st.Page("pages/chatbot.py")

pages = st.navigation(pages=[
    st.Page(page=introduction),
    classification_page,
    chatbot_page
])

pages.run()
