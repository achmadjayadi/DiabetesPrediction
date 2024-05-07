import pickle
import streamlit as st

# Read model
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

# Title web
st.title('Diabetes Prediction')

# Column
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.text_input('Input Nilai Pregnancies', key='Pregnancies')
    BloodPressure = st.text_input('Input Nilai BloodPressure', key='BloodPressure')
    SkinThickness = st.text_input('Input Nilai SkinThickness', key='SkinThickness')
    DiabetesPedigreeFunction = st.text_input('Input Nilai DiabetesPedigreeFunction', key='DiabetesPedigreeFunction')
with col2:
    Glucose = st.text_input('Input Nilai Glucose', key='Glucose')
    Insulin = st.text_input('Input Nilai Insulin', key='Insulin')
    BMI = st.text_input('Input Nilai BMI', key='BMI')
    Age = st.text_input('Input Age', key='Age')

# Code
diab_diagnosis = ''

# Button Prediction
if st.button('Tes Prediksi Diabetes'):
    diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    if diab_prediction[0] == 1:
        diab_diagnosis = 'Pasien Terkena Diabetes'
    else:
        diab_diagnosis = 'Pasien Tidak Terkena Diabetes'
    st.success(diab_diagnosis)
