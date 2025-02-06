import streamlit as st
import pdfplumber
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline(
    "question-answering", model="distilbert-base-uncased-distilled-squad"
)


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()


# Chunking text
def chunk_text(text, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    return splitter.split_text(text)


# Generate embeddings
def generate_embeddings(text_chunks):
    return embedding_model.encode(text_chunks, convert_to_numpy=True)


# Create FAISS index
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


# Retrieve relevant context (Increased context size)
def retrieve_context(query, index, text_chunks, top_k=7):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_text = "\n".join([text_chunks[i] for i in indices[0]])
    return retrieved_text


# Generate Answer
def answer_question(query, faiss_index, book_chunks):
    context = retrieve_context(query, faiss_index, book_chunks)
    result = qa_pipeline(question=query, context=context, max_answer_len=150)
    return result["answer"] + "\n\n**Additional Context:** " + context[:400] + "..."


# Streamlit UI
st.title("📖 Book-Based Question Answering System")
st.write("Upload a book (PDF) and ask any question!")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

if uploaded_file:
    st.write("Processing book...")
    book_text = extract_text_from_pdf(uploaded_file)
    book_chunks = chunk_text(book_text)
    chunk_embeddings = generate_embeddings(book_chunks)
    faiss_index = create_faiss_index(chunk_embeddings)
    st.success(f"Book processed successfully! ({len(book_chunks)} chunks)")

    query = st.text_input("Ask a question based on the book:")
    if query:
        answer = answer_question(query, faiss_index, book_chunks)
        st.write(f"**Answer:** {answer}")
