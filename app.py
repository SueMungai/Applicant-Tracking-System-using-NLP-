import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import scipy

# Load SpaCy model
import spacy
nlp = spacy.load('en_core_web_sm')

# Load pre-trained model and vectorizer
# Assuming you saved these models
import pickle
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Streamlit UI
st.title("Resume Scorer")
uploaded_resume = st.file_uploader("Upload Resume", type=["txt", "pdf", "docx"])
uploaded_job_description = st.file_uploader("Upload Job Description", type=["txt", "pdf", "docx"])

def extract_text(file):
    # Add code to handle different file types (txt, pdf, docx)
    return file.read().decode('utf-8')

if uploaded_resume and uploaded_job_description:
    resume_text = extract_text(uploaded_resume)
    job_description = extract_text(uploaded_job_description)

    cleaned_resume = clean_text(resume_text)
    cleaned_job_description = clean_text(job_description)

    # Transform using vectorizer
    resume_tfidf = vectorizer.transform([cleaned_resume])
    job_tfidf = vectorizer.transform([cleaned_job_description])
    X = scipy.sparse.hstack([resume_tfidf, job_tfidf])

    # Predict
    score = clf.predict_proba(X)[0][1]
    st.write(f"Resume Score: {score:.2f}")

    # Highlight missing keywords
    job_tokens = set(cleaned_job_description.split())
    resume_tokens = set(cleaned_resume.split())
    missing_keywords = job_tokens - resume_tokens
    st.write("Missing Keywords:", ", ".join(missing_keywords))
