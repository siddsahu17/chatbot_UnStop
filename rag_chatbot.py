import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os

# Define the path to the PDF file inside the "data" directory
PDF_FILE_PATH = "data/data.pdf"

# Load Environment Variables (Optional: Set API keys if needed)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")  # Set your OpenAI API key if required

# Load and Process PDF Data
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    return [chunk.page_content for chunk in chunks]

# Initialize Vector Store
def chunk_and_store_data(pdf_path):
    chunks = load_pdf(pdf_path)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Free alternative to OpenAI

    # Create and save FAISS vector store
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_index")

    return vector_store

# Load Vector Store
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Chatbot Function
def get_response(query):
    vector_store = load_vector_store()
    docs = vector_store.similarity_search(query, k=3)
    
    response_text = "\n\n".join([doc.page_content for doc in docs])
    return response_text

# Streamlit UI
def main():
    st.set_page_config(page_title="Employee Sentiment Analysis Chatbot", layout="wide")
    st.title("🤖 Employee Burnout & Mood Analysis Chatbot")

    # Chat Interface
    query = st.text_input("💬 Enter your query:")
    if query:
        response = get_response(query)
        st.write("🤖 Chatbot:", response)

if __name__ == "__main__":
    main()
