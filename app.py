import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load the pre-trained model and CountVectorizer
with open('pickle/spam_classifier.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('pickle/count_vectorizer.pkl', 'rb') as cv_file:
    cv = pickle.load(cv_file)

def clean_text(text):
    if text is not None and text.strip():  # Check if text is not None and not empty
        # remove html tags and non-alphabetic 
        text = re.sub('<.*?>', '', text)
        text = re.sub('[^a-zA-Z]', ' ', text).lower()
        words = nltk.word_tokenize(text)
        words = [w for w in words if w not in stopwords.words('english')]
        stemmer = PorterStemmer()
        words = [stemmer.stem(w) for w in words]
        # Join the words back into a string
        text = ' '.join(words)
        return text
    else:
        return ""

st.title('Identify Spam messages')

# Define CSS styles
style = """
<style>
    h1 {
        color: #ede5e5;
        text-align: center;
    }
    .button {
        background-color: #f96969;
        color: #f9f2f2;
        border: 10px;
        padding: 10px 20px;
        text-align: center;
        text-decoration: absolute;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
</style>
"""

st.markdown(style, unsafe_allow_html=True)

message = st.text_area('Enter a text to check:')
if st.button('Identify', key='classify_button'):
    cleaned_message = clean_text(message)
    message_vector = cv.transform([cleaned_message]).toarray()
    prediction = classifier.predict(message_vector)
    st.write(f'This message is identified as : {prediction[0]}')
    
st.text('© sam-2023 deployed. All rights reserved.')