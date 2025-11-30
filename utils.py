import requests
import tempfile
import os
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import csv
from datetime import datetime

def load_pdf(url):
    """Tenta carregar um PDF da URL. Se falhar, usa o caminho de fallback local."""
    abs_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(abs_path)

    fallback_path = os.path.join(dir_path,"data", "Novo.pdf")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        print("PDF carregado com sucesso da URL.")
        return tmp_file_path, 1
    except requests.exceptions.RequestException as e:
        print(f"Falha ao carregar da URL ({e}). Usando o arquivo de fallback: {fallback_path}")
        return fallback_path, 0


def split_pdf(tmp_file_path, is_url):
    try:
        loader = PyPDFLoader(tmp_file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            
        )
        splits = text_splitter.split_documents(data)
    finally:
        if is_url:
            os.remove(tmp_file_path)

    return splits
def format_docs(docs):
  return "\n\n".join([d.page_content for d in docs])

def save_score(question, answer, score, save_path, filename="faithfulness_scores.csv"):
    """Saves the faithfulness score to a CSV file."""
    # Garante que o diretório 'evaluation' exista
    output_dir = os.path.join(save_path, filename)
    #output_dir = os.path.dirname(filename)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_exists = os.path.isfile(output_dir)
    with open(output_dir, 'a', newline='', encoding='utf-8') as csvfile:
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