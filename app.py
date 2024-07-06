import streamlit as st
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline

# Load or train your best model (RandomForestClassifier in this case)
best_model_path = 'models/RandomForest_best_model.pkl'
with open(best_model_path, 'rb') as f:
    best_model = pickle.load(f)

# Load the preprocessor
preprocessor_path = 'models/preprocessor.pkl'
with open(preprocessor_path, 'rb') as f:
    preprocessor = pickle.load(f)

# Streamlit app UI
st.title('Mental Health Predictor')
st.image('misc/mental1.jpg', use_column_width=True)

st.sidebar.header('A quick survey to predict mental health')

# Collect user inputs
def collect_inputs():
    inputs = {}
    inputs['Age'] = st.sidebar.slider('What is you age?', 18, 100, 30)
    inputs['Gender'] = st.sidebar.radio('Gender', ['male', 'female', 'others'])
    inputs['family_history'] = st.sidebar.radio('Do you have any medical history of Mental health?', ['Yes', 'No'])
    inputs['no_employees'] = st.sidebar.radio('Number of Employees in your Company', ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000'])
    inputs['remote_work'] = st.sidebar.radio('Do you work Remotely?', ['Yes', 'No'])
    inputs['tech_company'] = st.sidebar.radio('Do you work in a Tech company?', ['Yes', 'No'])
    inputs['benefits'] = st.sidebar.radio('Does your employer provide mental health benefits?', ['Yes', 'No', 'Don\'t know'])
    inputs['care_options'] = st.sidebar.radio('Are you aware of the mental health care options available to you?', ['Yes', 'No', 'Not sure'])
    inputs['wellness_program'] = st.sidebar.radio('Does your employer offer a wellness program?', ['Yes', 'No', 'Don\'t know'])
    inputs['seek_help'] = st.sidebar.radio('Do you know where to seek help for mental health issues?', ['Yes', 'No', 'Don\'t know'])
    inputs['anonymity'] = st.sidebar.radio('Do you feel your privacy is protected if you seek mental health treatment?', ['Yes', 'No', 'Don\'t know'])
    inputs['leave'] = st.sidebar.radio('How easy is it for you to take medical leave for mental health reasons?', ['Somewhat easy', 'Somewhat difficult', 'Very easy', 'Very difficult', 'Don\'t know'])
    inputs['mental_health_consequence'] = st.sidebar.radio('Could discussing mental health issues at work have negative consequences?', ['Yes', 'No', 'Maybe'])
    inputs['phys_health_consequence'] = st.sidebar.radio('Could discussing physical health issues at work have negative consequences?', ['Yes', 'No', 'Maybe'])
    inputs['coworkers'] = st.sidebar.radio('Are you comfortable discussing mental health issues with your coworkers?', ['Yes', 'No', 'Some of them'])
    inputs['supervisor'] = st.sidebar.radio('Are you comfortable discussing mental health issues with your supervisor?', ['Yes', 'No', 'Some of them'])
    inputs['mental_health_interview'] = st.sidebar.radio('Has mental health ever been discussed during your job interviews?', ['Yes', 'No', 'Maybe'])
    inputs['phys_health_interview'] = st.sidebar.radio('Has physical health ever been discussed during your job interviews?', ['Yes', 'No', 'Maybe'])
    inputs['mental_vs_physical'] = st.sidebar.radio('Does your employer address mental health as much as physical health?', ['Yes', 'No', 'Don\'t know'])
    inputs['obs_consequence'] = st.sidebar.radio('Do you think mental health issues affect your job performance?', ['Yes', 'No'])
    return pd.DataFrame([inputs])

user_input = collect_inputs()

# Ensure columns are in the same order as during training
expected_columns = ['Age', 'Gender', 'family_history', 'no_employees', 'remote_work', 'tech_company', 'benefits',
                    'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence',
                    'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                    'mental_vs_physical', 'obs_consequence']
user_input = user_input[expected_columns]
print(f'user_input.shape = {user_input.shape}')

# Preprocess user input
user_input_preprocessed = preprocessor.transform(user_input)
print(f'user_input_preprocessed.shape = {user_input_preprocessed.shape}')

if st.sidebar.button('Predict'):
    prediction = best_model.predict(user_input_preprocessed)
    prediction_proba = best_model.predict_proba(user_input_preprocessed)
    print(f'prediction = {prediction}')
    
    st.subheader('Prediction')
    if prediction[0] == 1:
        st.write("Hello! From your responses, it appears that consulting a doctor might be beneficial. Don't worry; we are here for you every step of the way. With the right support, you'll be back on track soon! Take care!")
    else:
        st.write("Hi! Based on your responses, it seems like you are doing well. Keep it up!")
        st.balloons()
    st.subheader('Prediction Probability')
    st.write(f"Probability of needing treatment: {prediction_proba[0][1]:.2f}")
    st.write(f"Probability of not needing treatment: {prediction_proba[0][0]:.2f}")

    