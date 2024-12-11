import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
import requests
from io import StringIO

# Set Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# URL of the CSV file in your GitHub repository (raw file URL)
csv_url = "https://raw.githubusercontent.com/AbhijeetStudies/chat/main/Possible_case_Preprocessing_Small.csv"

# Function to download the CSV file from GitHub
def load_data_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        # Use StringIO to read the CSV from the text content
        return pd.read_csv(StringIO(response.text))
    else:
        st.error("Failed to load data from the provided URL.")
        return None

# Load the CSV file from GitHub
df = load_data_from_github(csv_url)

if df is not None:
    # Rest of the processing code
    embedding_model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")
    df['combined'] = df['human_clean'] + " " + df['gpt_clean']
    
    embeddings = np.array([embedding_model.encode(text) for text in df['combined']])
    
    # FAISS index creation
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # IVF Index creation
    nlist = 20
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    index.train(embeddings)
    index.add(embeddings)

    # Search function
    def search_faiss(query, k=10):
        query_embedding = embedding_model.encode(query).reshape(1, -1)
        D, I = index.search(query_embedding, k)
        results = [df.iloc[i] for i in I[0]]
        return results

    # Load the FLAN-T5 model
    llm_model_name = "google/flan-t5-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSeq2SeqLM.from_pretrained(
        llm_model_name,
        device_map="auto" if device == "cuda" else "cpu",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=500, num_beams=2, num_return_sequences=1, temperature=0.4)

    def generate_response(query, k=5):
        retrieved_results = search_faiss(query, k)
        context = "\n".join([f"User said: {result['human_clean']}\nResponse: {result['gpt_clean']}" for result in retrieved_results[:3]])
        prompt = f"Context:\n{context}\n\nQuery: {query}\n\nAs an empathetic assistant, consider the user's situation and the context provided above. Respond with detailed and actionable advice that addresses their concerns thoughtfully."
        response = llm(prompt)[0]['generated_text']
        return response

    # Streamlit UI
    st.title("Empathetic Chatbot for Stress Management")
    query = st.text_input("Ask a question about stress management:")

    if query:
        st.write("Generating response...")
        response = generate_response(query)
        st.write(f"Chatbot Response: {response}")
