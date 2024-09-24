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


# Define category mapping
category_mapping = {
    1: "World News",
    2: "Sports",
    3: "Business",
    4: "Technology"
}

# Streamlit UI
st.title("News Article Classifier")

user_input = st.text_area("Enter a news article description:")

if st.button("Classify"):
    if user_input:
        clean_input = preprocess_text(user_input)
        input_vector = vectorizer.transform([clean_input])
        
        # Predict the class
        predicted_class = model.predict(input_vector)[0]
        
        # Map the class index to category label
        category_label = category_mapping.get(predicted_class, "Unknown Category")
        
        # Display the result
        st.write(f"The predicted class for this article is: {category_label}")
    else:
        st.write("Please enter a description.")

