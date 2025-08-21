import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import io
import re

# App config
st.set_page_config(page_title="🔍 Smart Resume Matcher", layout="centered")
st.markdown("<h1 style='text-align: center; color: teal;'>🧠 AI Resume Matcher</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Match resumes to job descriptions with smart AI embeddings</p>", unsafe_allow_html=True)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        return "".join([page.extract_text() or "" for page in pdf.pages])

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# Upload section
st.subheader("📄 Upload Job Description")
jd_file = st.file_uploader("Upload a single Job Description (PDF)", type=["pdf"])

st.subheader("📁 Upload Resumes")
resumes = st.file_uploader("Upload one or more Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

# Match button
if st.button("🔎 Match Resumes") and jd_file and resumes:
    jd_text = clean_text(extract_text_from_pdf(jd_file))
    jd_embedding = model.encode([jd_text])

    results = []
    for resume in resumes:
        resume_text = clean_text(extract_text_from_pdf(resume))
        if not resume_text.strip():
            st.warning(f"⚠️ {resume.name} appears to be empty or unreadable.")
            continue

        resume_embedding = model.encode([resume_text])
        score = np.dot(jd_embedding, resume_embedding.T)[0]
        results.append({"Resume": resume.name, "Match Score": round(float(score), 4)})

    if results:
        df = pd.DataFrame(results).sort_values(by="Match Score", ascending=False)
        st.success("✅ Matching Complete!")
        st.dataframe(df, use_container_width=True)

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, "resume_match_results.csv", "text/csv")
    else:
        st.error("❌ No valid resumes found.")
elif st.button("🔎 Match Resumes"):
    st.warning("📌 Please upload both Job Description and Resumes.")
