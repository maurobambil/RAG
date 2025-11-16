import requests
import tempfile
import os
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(url):
    response = requests.get(url)
    response.raise_for_status() 
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name

    return tmp_file_path

def verify_collection(collection_name, client):
    try:
        collection = client.get_collection(name=collection_name)
        if collection.count() > 0:
            print(f"Collection '{collection_name}' already contains documents. Skipping upload.")
            return  0# Exit the main function
    except ValueError:
        print(f"Collection '{collection_name}' does not exist. Proceeding with creation.")
    return 1

def split_pdf(tmp_file_path):
    try:
        loader = PyPDFLoader(tmp_file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=200,
        )
        splits = text_splitter.split_documents(data)
    finally:
        os.remove(tmp_file_path)

    return splits
