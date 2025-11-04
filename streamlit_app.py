import streamlit as st
import os
from pdfplumber import open as pdf_open
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load models
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    mistral_model = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(mistral_model)
    model_mistral = AutoModelForCausalLM.from_pretrained(
        mistral_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    qa_pipeline = pipeline("text-generation", model=model_mistral, tokenizer=tokenizer, max_new_tokens=128)
    return embedder, qa_pipeline

embedder, qa_pipeline = load_models()

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdf_open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_texts(chunks):
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    return np.array(embeddings, dtype="float32")

def search(query, chunks, embeddings, top_k=3):
    q_embed = embedder.encode([query])
    sims = cosine_similarity(q_embed, embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

st.title("PDF Q&A with Mistral LLM")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)
        embeddings = embed_texts(chunks)
        st.success("PDF processed and indexed!")

    query = st.text_input("Ask a question about the PDF:")
    if query:
        context_chunks = search(query, chunks, embeddings)
        context = "\n\n".join(context_chunks)
        prompt = f"Answer the following question as concisely as possible based on the provided PDF content.\n\nPDF Content:\n{context}\n\nQuestion: {query}\nAnswer:"
        with st.spinner("Generating answer with Mistral..."):
            answer = qa_pipeline(prompt)[0]["generated_text"][len(prompt):].strip()
        st.markdown(f"**Answer:** {answer}")
