from dotenv import load_dotenv
import requests
import tempfile
import os
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import csv
from datetime import datetime

def load_pdf(url):
    response = requests.get(url)
    response.raise_for_status() 
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name

    return tmp_file_path

def split_pdf(tmp_file_path):
    try:
        loader = PyPDFLoader(tmp_file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            
        )
        splits = text_splitter.split_documents(data)
    finally:
        os.remove(tmp_file_path)

    return splits
def format_docs(docs):
  return "\n\n".join([d.page_content for d in docs])

def save_score(question, answer, score, filename="evaluation/faithfulness_scores.csv"):
    """Saves the faithfulness score to a CSV file."""
    # Garante que o diretório 'evaluation' exista
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'question', 'answer', 'faithfulness_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'faithfulness_score': score
        })