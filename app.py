import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import io

# Set page config
st.set_page_config(page_title="Smart Resume Matcher", layout="centered")

# Title
st.title("🤖 Smart Resume Matcher")

# Load the model (CPU only)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

model = load_model()

# Function to extract text from PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(io.BytesIO(file.read())) as pdf:
        return " ".join([page.extract_text() or "" for page in pdf.pages])

# Function to clean and extract keywords
def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = {"and", "or", "with", "in", "on", "the", "a", "an", "to", "of", "for",
                 "we", "you", "are", "is", "looking", "need", "have", "has"}
    return set(w for w in words if w not in stopwords and len(w) > 2)

# Upload UI
jd_file = st.file_uploader("📄 Upload Job Description (PDF)", type=["pdf"])
resumes = st.file_uploader("📁 Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)

# Match button
if st.button("🔎 Match Resumes"):
    if jd_file and resumes:
        jd_text = extract_text_from_pdf(jd_file)
        jd_embedding = model.encode([jd_text])
        jd_keywords = extract_keywords(jd_text)

        results = []

        for resume in resumes:
            resume_text = extract_text_from_pdf(resume)
            if not resume_text.strip():
                st.warning(f"⚠️ {resume.name} is empty or unreadable.")
                continue

            resume_embedding = model.encode([resume_text])
            score = np.dot(jd_embedding, resume_embedding.T)[0]
            resume_keywords = extract_keywords(resume_text)
            matched_keywords = ", ".join(sorted(jd_keywords.intersection(resume_keywords)))

            results.append({
                "Resume": resume.name,
                "Match Score": round(float(score), 4),
                "Matched Keywords": matched_keywords
            })

        if results:
            df = pd.DataFrame(results).sort_values(by="Match Score", ascending=False)
            st.success("✅ Matching Complete!")
            st.dataframe(df, use_container_width=True)

            # CSV download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, "resume_match_results.csv", "text/csv")
        else:
            st.error("❌ No valid resumes found.")
    else:
        st.warning("📌 Please upload both the Job Description and at least one Resume.")
