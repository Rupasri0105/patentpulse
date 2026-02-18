
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load saved files
df = pd.read_csv("uspto_abstracts.csv")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")

st.title("PatentPulse 🔎")
st.subheader("AI-Based Patent Similarity & Fraud Detection")

# Text input
user_input = st.text_area("Enter Patent Abstract:")

# Analyze button
if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Please enter some text first.")

    else:
        # Convert input to vector
        user_vector = vectorizer.transform([user_input])

        # Compute similarity
        similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

        # Get top 5 similar patents
        top_indices = similarity_scores.argsort()[::-1][:5]

        max_similarity = np.max(similarity_scores)
        novelty_score = 1 - max_similarity

        st.write("### 🔎 Top 5 Similar Patents")

        for idx in top_indices:
            st.write("Patent ID:", df.iloc[idx]["patent_id"])
            st.write("Similarity:", round(float(similarity_scores[idx]), 4))
            st.write("---")

        st.write("### 📊 Novelty Score:", round(float(novelty_score), 4))

        # Fraud detection condition
        if max_similarity > 0.85:
            st.error("⚠ Potential Duplicate / Fraud Detected")
        else:
            st.success("✅ Patent Appears Novel")
