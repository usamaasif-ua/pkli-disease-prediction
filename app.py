import streamlit as st
import pandas as pd
import joblib
import re

# Load the model and label encoder
model = joblib.load('improved_medical_diagnosis_model.joblib')
le = joblib.load('label_encoder.joblib')

# Function to preprocess text input
def preprocess_text(text):
    text = str(text).lower()  # Lowercase and ensure string type
    text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters
    return text

# Streamlit UI
st.title("Disease Prediction System")
st.write("Input the patient's information below:")

# Input fields
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.number_input("Age", min_value=18, max_value=80)
medical_history = st.text_area("Medical History")

# Predict button
if st.button("Predict Disease"):
    if medical_history:
        # Preprocess inputs
        gender_encoded = 0 if gender == 'Male' else 1
        preprocessed_history = preprocess_text(medical_history)
        
        # Prepare the input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender_encoded],
            'Medical History': [preprocessed_history]
        })
        
        # Make prediction and get prediction probabilities
        prediction_proba = model.predict_proba(input_data)
        max_proba = prediction_proba.max()
        
        # Only make prediction if confidence is greater than 60%
        if max_proba >= 0.60:
            prediction = model.predict(input_data)
            predicted_disease = le.inverse_transform(prediction)
            st.success(f"The predicted disease is: {predicted_disease[0]} with {max_proba*100:.2f}% confidence.")
        else:
            st.warning("The model is unsure about the disease. Please provide more details.")
    else:
        st.error("Please enter the medical history.")
