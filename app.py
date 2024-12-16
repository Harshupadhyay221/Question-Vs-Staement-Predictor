import streamlit as st
import pickle
import numpy as np

# Load the model and vectorizer
try:
    with open('logistic_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")

# Streamlit UI
st.title("Question or Statement Predictor")
st.write("Enter a sentence below:")

# Input text box
input_text = st.text_area("Input Sentence")

if st.button("Predict"):
    if input_text:
        # Vectorize the input text
        input_vector = vectorizer.transform([input_text])
        
        # Make prediction
        prediction = model.predict(input_vector)
        
        # Display the result
        if prediction[0] == 1:  # Assuming '1' is for 'question'
            st.success("The input is a Question.")
        else:
            st.success("The input is a Statement.")
    else:
        st.error("Please enter a sentence.")