# News Article Classification using Naive Bayes
This project classifies **News articles** into different categories using the **Naive Bayes classifier**. The model is built using **Scikit-learn** for machine learning, and the **TF-IDF Vectorizer** for transforming text data. The app is deployed using **Streamlit** to create an interactive user interface.

## Project Overview
The goal of this project is to classify news articles based on their content into predefined categories such as:

+ Sports
+ Politics
+ Technology
+ Business

## Features

+ Preprocessing of raw news article text.
+ Text vectorization using TF-IDF.
+ Classification using the Naive Bayes algorithm.
+ User-friendly web app interface built with Streamlit.

## Dataset
The dataset contains labeled news articles, which are used to train the model. The model predicts the category of a new article based on its description or content.
[LINK](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) For Dataset

## Model Evaluation
+ Accuracy: **88%**

### Classification Report:

![image](https://github.com/user-attachments/assets/cb1a9138-b45a-4642-804a-f1fbcd2b54bd)


### Confusion Matrix:

![image](https://github.com/user-attachments/assets/0423178d-d8c2-4cf6-b3e0-c5c9ae120e85)


## File Structure
+ **app.py:** Streamlit app that provides the user interface for classifying news articles.
+ **naive_bayes_model.pkl:** Pre-trained Naive Bayes model saved as a pickle file.
+ **tfidf_vectorizer.pkl:** Pre-trained TF-IDF vectorizer.
+ **requirements.txt:** File containing the necessary Python libraries to run the project.

## Streamlit App
The Streamlit app provides an interactive interface where users can input a news article's description or content, and the model will predict the article's category.

![image](https://github.com/user-attachments/assets/3ae3edb7-73bd-4176-9221-e9f29807c5dd)


## Deployment
The app can be deployed on platforms like Streamlit Cloud or Heroku for public access.

## Technologies Used
+ **Python:** Programming language.
+ **Scikit-learn:** Machine learning library.
+ **NLTK:** Natural language processing toolkit for text tokenization.
+ **Streamlit:** Framework for building interactive web apps.
+ **Pandas:** Data manipulation library.

## Contributing
Feel free to submit pull requests or open issues to improve the project.
