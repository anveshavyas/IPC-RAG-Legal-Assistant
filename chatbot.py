import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Load IPC data
df = pd.read_csv("C:\\Users\\dell\\Desktop\\chatbot\\ipc_sections.csv")
df["combined"] = df[["Section", "Description", "Offense", "Punishment"]].fillna("").agg(" ".join, axis=1)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["combined"])

# Streamlit setup
st.set_page_config(page_title="Legal RAG Chatbot", page_icon="⚖")
st.title("⚖ IPC RAG Legal Assistant")
user_input = st.text_input("🔍 Ask your legal question:")

if user_input:
    # Retrieve relevant context
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    top_idx = similarity.argmax()
    best_match = df.iloc[top_idx]
    retrieved_text = best_match["combined"]

    # Generate answer using Flan-T5
    generator = pipeline("text2text-generation", model="google/flan-t5-small", max_length=512)
    prompt = f"""Answer this legal query based on Indian Penal Code:

Query: "{user_input}"

Legal Context:
{retrieved_text}

Answer:"""

    response = generator(prompt)[0]["generated_text"]

    # Show Results
    st.subheader("📘 Closest IPC Match:")
    st.markdown(f"*Section:* {best_match['Section']}")
    st.markdown(f"*Description:* {best_match['Description']}")
    st.markdown(f"*Offense:* {best_match['Offense']}")
    st.markdown(f"*Punishment:* {best_match['Punishment']}")

    st.subheader("🧠 RAG Answer:")
    st.write(response)