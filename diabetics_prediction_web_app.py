# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 09:56:55 2024

@author: Sandeep
"""

import numpy as np
import pickle 
import streamlit as st

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def diabetic_prediction(inputx):
    input_ = np.asarray(inputx)
    input_ = input_.reshape(1, -1)
    prediction = loaded_model.predict(input_)
    if prediction[0]==0:
        return "Non-Diabetic"
    else:
        return 'Diabetic'
    
def main():
    st.title("Diabetic Prediction Web App")
    
    Pregnancies = st.text_input("Number of Pregnancies:")
    Glucose = st.text_input("Glucose Level:")
    BloodPressure = st.text_input("Blood Pressure:")
    SkinThickness = st.text_input("SkinThickness:")
    Insulin = st.text_input("Insulin Level:")
    BMI = st.text_input("BMI:")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function:")
    Age = st.text_input("Age:")
    
    diagnosis = ''
    
    if st.button('Predict'):
        diagnosis = diabetic_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
    
    
    
if __name__ == "__main__":
    main()
