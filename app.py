import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import scipy
import pdfplumber 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
import nltk
from nltk.corpus import stopwords

# Streamlit UI
st.title("Candidate Selection Tool")
st.subheader("NLP Based Resume Screening")
st.caption("Aim of this project is to check whether a candidate is qualified for a role based on the information captured on their resume. It's a form of pattern matching between a job's requirements and the qualifications of a candidate based on their resume.")
uploaded_resume = st.file_uploader("Upload Resume", type=["txt", "pdf", "docx"], accept_multiple_files=True)
job_description = st.text_area("Enter Job Description")
click = st.button("Match")


# # Load pre-trained model and vectorizer
# import pickle
# with open('vectorizer.pkl', 'rb') as f:
#     vectorizer = pickle.load(f)
# with open('classifier.pkl', 'rb') as f:
#     clf = pickle.load(f)

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean text 
def clean_text(text):
    # text = text.lower()
    text = re.sub(r'\s+', ' ', text) # to remove extra white space
    text = re.sub(r'[^\w\s]', '', text) # to remove all non-word characters 
    text = re.sub(r'[^\x00-\x7f]',r' ', text) # to remove anything that is not within the ASCII range 
    words = text.split()
    words = [word for word in words if word not in stop_words]
    cleaned_text = ' '.join(words)
    return text

# Function to extract text from pdf
def extract_txt(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# to make sure the uploaded resume is processed
if uploaded_resume is not None:
    resume = extract_txt(uploaded_resume)

def result(JD_txt, resume_txt):
    cleaned_jd = clean_text(JD_txt)
    cleaned_resume = clean_text(resume_txt)
    content = [JD_txt, resume_txt]
    # cv = CountVectorizer()
    # matrix = cv.fit_transform(content)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(content)
    similarity_matrix = cosine_similarity(matrix)
    score = similarity_matrix[0][1] * 100
    # resume_tfidf = vectorizer.transform([cleaned_resume])
    # job_tfidf = vectorizer.transform([cleaned_jd])
    # X = scipy.sparse.hstack([resume_tfidf, job_tfidf])

    # score = clf.predict_proba(X)[0][1] * 100
    return score

# Function to find missing keywords
def find_missing_keywords(JD_txt, resume_txt):
    job_tokens = set(clean_text(JD_txt).split())
    resume_tokens = set(clean_text(resume_txt).split())
    missing_keywords = job_tokens - resume_tokens
    return missing_keywords

# Processing resume and displaying results
if click:
    if job_description and uploaded_resume:
        resume_text = extract_txt(uploaded_resume)
        
        # Calculate match percentage
        match = result(job_description, resume_text)
        match = round(match, 2)
        st.write(f"Match Percentage: {match}%")
        
        # Find and display missing keywords
        missing_keywords = find_missing_keywords(job_description, resume_text)
        st.write("Missing Keywords:", ", ".join(missing_keywords))
    # else:
    #     st.write("Please enter a job description and upload a resume PDF.")

