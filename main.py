from utils import load_pdf, split_pdf, verify_collection
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

client = chromadb.CloudClient(
    api_key='ck-J8ft6fJfZF814R3NyFBP2Zjn5UJACqafWGuHTu4Hucgp',
    tenant='0b13fde1-bc33-4a4e-a07f-352107c7483e',
    database='rag_regulamento'
)
def main():
    collection_name = "pdf_embeddings"
    model = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")

    url = "https://www.facom.ufms.br/wp-content/uploads/2023/07/20232-EM-DIANTE-ATUAL-RESOLUCAO__COPP__n_704__de_07_07_2023.pdf"
    print(f"Fetching PDF from {url}...")
    tmp_path = load_pdf(url)
    print(f"split the pdf into chunks")
    splits = split_pdf(tmp_path)
    if verify_collection(collection_name, client):
        vectorstore = Chroma.from_documents(documents=splits, embedding=model, client=client, collection_name=collection_name)
        retriever = vectorstore.as_retriever(kwargs={"K": 5})
    else: 
        #TODO se ja tiver tem q fazer uma query, ou um retriever n sei como q é, nem sei se ta certo até aqui, fui fazendo até dar o upload no chroma tem q ver o q faz obg,
        #tem que ver tbm se vai continuar pegando da url, se vai usar só o documento mais novo ou os 2 q tem de 2019 e 2023
        # deixar mais bonito e "profissional, mas siso n sei fazer hehe"
        #tinha q da um jeito de automatiar a apikey pq tem algum warning falando q n é seguro deixar a chave no codigo kkkkkkk
    

  
if __name__ == "__main__":
    main()
    
    