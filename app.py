import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')

# Load the trained model
model = joblib.load('naive_bayes_model.pkl')  # Save your model using joblib

# Load the TF-IDF Vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Save your vectorizer using joblib

# Preprocess function without NLTK
def preprocess_text(text):
    # Convert to lowercase and remove non-alphabetic characters
    tokens = text.lower().split()
    tokens = [word for word in tokens if word.isalpha()]
    return " ".join(tokens)


# Streamlit App
st.title("News Article Classifier")

# Input from the user
user_input = st.text_area("Enter a news article description for classification:")

if st.button('Classify'):
    if user_input:
        # Preprocess the input
        clean_input = preprocess_text(user_input)
        
        # Vectorize the input
        input_vector = vectorizer.transform([clean_input])
        
        # Make a prediction
        prediction = model.predict(input_vector)[0]
        
        # Display the result
        st.write(f"The predicted class for this article is: {prediction}")
    else:
        st.write("Please enter a news article description.")
