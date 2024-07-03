import streamlit as st
import docx2txt
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Streamlit UI
st.title("Resume Matching and Screening Tool")
st.subheader("Upload your resumes and provide a job description to find the best matches.")
st.caption("The aim of this project is to check whether a candidate is qualified for a role based on the information captured on their resume.  ")

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # to remove extra white space
    text = re.sub(r'[^\w\s]', '', text)  # to remove all non-word characters
    text = re.sub(r'[^\x00-\x7f]', r' ', text)  # to remove anything that is not within the ASCII range
    words = text.split()
    words = [word for word in words if word not in stop_words]
    cleaned_text = ' '.join(words)
    return cleaned_text

# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    return docx2txt.process(file)

# Function to extract text from TXT
def extract_text_from_txt(file):
    return file.read().decode('utf-8')

# Function to extract text based on file extension
def extract_text(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file)
    elif file.type == "text/plain":
        return extract_text_from_txt(file)
    else:
        return ""

# Function to calculate resume score
def result(JD_txt, resume_txt):
    cleaned_jd = clean_text(JD_txt)
    cleaned_resume = clean_text(resume_txt)
    content = [cleaned_jd, cleaned_resume]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(content)
    similarity_matrix = cosine_similarity(matrix)
    score = similarity_matrix[0][1] * 100
    return score

# We want to be ablt to Upload multiple resumes
uploaded_resumes = st.file_uploader("Upload Resumes", type=["txt", "pdf", "docx"], accept_multiple_files=True)

# Enter job description
job_description = st.text_area("Enter Job Description")

# Match resumes button
if st.button("Match"):
    if uploaded_resumes and job_description:
        resumes = [extract_text(file) for file in uploaded_resumes]

        # Calculate match scores
        scores = [result(job_description, resume) for resume in resumes]

        # Get top 5 resumes and their scores
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        top_resumes = [uploaded_resumes[i].name for i in top_indices]
        resume_scores = [round(scores[i], 2) for i in top_indices]

        st.write("Top matching resumes:")
        for i in range(len(top_resumes)):
            st.write(f"{i + 1}. {top_resumes[i]} with a score of {resume_scores[i]}%")
    else:
        st.error("Please upload resumes and enter a job description.")
