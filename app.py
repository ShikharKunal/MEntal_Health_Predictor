import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder

# Load the encoders and the model
with open('ord_encoder.pkl', 'rb') as f:
    ord_encoder = pickle.load(f)

with open('std_scaler.pkl', 'rb') as f:
    std_scaler = pickle.load(f)

with open('XGBoost_final.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the columns and options for the inputs
gender_options = ['Female', 'Male', 'Other']
self_employed_options = ['No', 'Yes']
family_history_options = ['No', 'Yes']
work_interfere_options = ['Never', 'Rarely', 'Sometimes', 'Often']
no_employees_options = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']
remote_work_options = ['No', 'Yes']
tech_company_options = ['No', 'Yes']
benefits_options = ['No', "Don't know", 'Yes']
care_options = ['No', 'Not sure', 'Yes']
wellness_program_options = ['No', "Don't know", 'Yes']
seek_help_options = ['No', "Don't know", 'Yes']
anonymity_options = ['No', "Don't know", 'Yes']
leave_options = ['Very easy', 'Somewhat easy', "Don't know", 'Somewhat difficult', 'Very difficult']
mental_health_consequence_options = ['No', 'Maybe', 'Yes']
phys_health_consequence_options = ['No', 'Maybe', 'Yes']
coworkers_options = ['No', 'Some of them', 'Yes']
supervisor_options = ['No', 'Some of them', 'Yes']
mental_health_interview_options = ['No', 'Maybe', 'Yes']
phys_health_interview_options = ['No', 'Maybe', 'Yes']
mental_vs_physical_options = ["Don't know", 'No', 'Yes']
obs_consequence_options = ['No', 'Yes']

# Streamlit UI
st.title("Mental Health Prediction App")

st.sidebar.title("A Survey to Predict Mental Health in Tech")

age = st.sidebar.slider("Age", 15, 75, 25)
gender = st.sidebar.radio("Gender", gender_options)
self_employed = st.sidebar.radio("Self-employed", self_employed_options)
family_history = st.sidebar.radio("Family History of Mental Illness", family_history_options)
work_interfere = st.sidebar.radio("Work Interference", work_interfere_options)
no_employees = st.sidebar.radio("Number of Employees", no_employees_options)
remote_work = st.sidebar.radio("Remote Work", remote_work_options)
tech_company = st.sidebar.radio("Tech Company", tech_company_options)
benefits = st.sidebar.radio("Benefits", benefits_options)
care_options = st.sidebar.radio("Care Options", care_options)
wellness_program = st.sidebar.radio("Wellness Program", wellness_program_options)
seek_help = st.sidebar.radio("Seek Help", seek_help_options)
anonymity = st.sidebar.radio("Anonymity", anonymity_options)
leave = st.sidebar.radio("Leave", leave_options)
mental_health_consequence = st.sidebar.radio("Mental Health Consequence", mental_health_consequence_options)
phys_health_consequence = st.sidebar.radio("Physical Health Consequence", phys_health_consequence_options)
coworkers = st.sidebar.radio("Coworkers", coworkers_options)
supervisor = st.sidebar.radio("Supervisor", supervisor_options)
mental_health_interview = st.sidebar.radio("Mental Health Interview", mental_health_interview_options)
phys_health_interview = st.sidebar.radio("Physical Health Interview", phys_health_interview_options)
mental_vs_physical = st.sidebar.radio("Mental vs Physical Health", mental_vs_physical_options)
obs_consequence = st.sidebar.radio("Observed Consequence", obs_consequence_options)

# Collect the user inputs into a DataFrame
data = {
    'age': [age],
    'gender': [gender],
    'self_employed': [self_employed],
    'family_history': [family_history],
    'work_interfere': [work_interfere],
    'no_employees': [no_employees],
    'remote_work': [remote_work],
    'tech_company': [tech_company],
    'benefits': [benefits],
    'care_options': [care_options],
    'wellness_program': [wellness_program],
    'seek_help': [seek_help],
    'anonymity': [anonymity],
    'leave': [leave],
    'mental_health_consequence': [mental_health_consequence],
    'phys_health_consequence': [phys_health_consequence],
    'coworkers': [coworkers],
    'supervisor': [supervisor],
    'mental_health_interview': [mental_health_interview],
    'phys_health_interview': [phys_health_interview],
    'mental_vs_physical': [mental_vs_physical],
    'obs_consequence': [obs_consequence],
}

df_input = pd.DataFrame(data)

# Separate age from the other features for encoding and scaling
age_column = df_input[['age']]
categorical_columns = df_input.drop(columns=['age'])

# Apply the encoders
categorical_columns_encoded = ord_encoder.transform(categorical_columns)
df_input_encoded = pd.DataFrame(categorical_columns_encoded, columns=categorical_columns.columns)

# Combine the age column with the encoded and scaled features
df_input_combined = pd.concat([age_column, df_input_encoded], axis=1)
df_input_combined_scaled = std_scaler.transform(df_input_combined)

st.image('misc/mental1.jpg',use_column_width=True)

# Make prediction
if st.button("Predict",use_container_width=True):
    prediction = model.predict(df_input_combined_scaled)
    prediction_proba = model.predict_proba(df_input_combined_scaled)

    if prediction[0] == 1:
        st.write("Prediction: **Requires Treatment**")
    else:
        st.write("Prediction: **Does not require Treatment**")

    st.write("Prediction Probability:")
    st.write(f"Does not require Treatment: {prediction_proba[0][0]:.2f}")
    st.write(f"Requires Treatment: {prediction_proba[0][1]:.2f}")
