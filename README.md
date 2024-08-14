# Applicant Tracking System using NLP

## Business Understanding

### Overview
The recruitment process is a critical function of Human Resources (HR) departments, tasked with identifying and hiring the most suitable candidates for various roles within an organization. With an increasing number of job applications, HR professionals face challenges in efficiently and effectively screening resumes to find top talent. An automated Applicant Tracking System (ATS) that leverages Natural Language Processing (NLP) can significantly enhance this process by matching candidate resumes with job descriptions, thus streamlining the selection process and ensuring the best fit for the role.

### Objective

The objective of this project is to develop an ATS that can evaluate and rank candidate resumes based on their relevance to a given job description. The system will allow HR professionals to upload multiple resumes and provide a job description, and it will output the top five candidates who best match the job requirements. This tool aims to improve the efficiency and accuracy of the recruitment process.

### Problem Statement

HR departments receive a large volume of applications for each job posting, making it time-consuming and challenging to manually review each resume and identify the most qualified candidates. The manual process is prone to human error and biases, which can result in overlooking suitable candidates. There is a need for an automated solution that can quickly and accurately match resumes with job descriptions, highlighting the top candidates for further consideration.

### Objectives

- Develop an NLP-based ATS that can process and analyze resumes in various formats (PDF, DOCX, TXT).
- Implement a system to clean and preprocess text data from resumes and job descriptions.
- Use TF-IDF vectorization to convert text data into numerical representations.
- Calculate the cosine similarity between job descriptions and resumes to determine the match score.
- Rank and display the top five candidates based on their match scores.

## Data Understanding

Input Data
Resumes: The resumes will be uploaded in multiple formats (PDF, DOCX, TXT). Each resume contains unstructured text with information about the candidate's skills, experience, education, and other relevant details.
Job Description: The job description is entered as plain text, detailing the qualifications, skills, and experience required for the role.
Data Extraction and Cleaning
Text Extraction: Depending on the file format, appropriate libraries (pdfplumber, docx2txt) are used to extract text content from resumes.
Text Cleaning: The extracted text is cleaned by removing non-word characters, extra white spaces, and stop words using regular expressions and NLTK's stop words list.

## Modeling

- Text Vectorization
  To compare the job description and resumes, we convert the text data into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. This technique helps in understanding the importance of words in the context of the documents.

- Similarity Measurement
Cosine similarity is used to measure the similarity between the job description vector and each resume vector. The cosine similarity score ranges from 0 to 1, where 1 indicates a perfect match.

- Ranking
  The resumes are ranked based on their cosine similarity scores with the job description. The top five resumes with the highest scores are selected and displayed.

## Implementation

#### Streamlit Application

The application is built using Streamlit, which provides an interactive web interface for uploading resumes, entering job descriptions, and displaying results. The key steps in the implementation are:

File Upload: Users can upload multiple resumes in different formats.

Job Description Input: Users enter the job description in a text area.

Text Extraction and Cleaning: Text is extracted from resumes and cleaned.

Match Calculation: The similarity between the job description and each resume is calculated using TF-IDF and cosine similarity.

Result Display: The top five matching resumes and their scores are displayed.
