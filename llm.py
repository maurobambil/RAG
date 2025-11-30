from utils import load_pdf, split_pdf, format_docs, save_score
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import re

def load_and_split(url):
    print(f"Carregando o PDF da URL: {url}...")
    tmp_path, is_url = load_pdf(url)
    print("Dividindo o PDF em partes...")
    splits = split_pdf(tmp_path, is_url)
    return splits

class ScoreParser(BaseOutputParser):
    def parse(self, text: str) -> float:
        try:
            return float(text.strip())
        except ValueError:
            match = re.search(r'\d+(\.\d+)?', text)
            if match:
                return float(match.group(0))
            return 0.0

def create_factscore_chain():
    """Cria uma cadeia para pontuar a fidelidade de uma resposta ao seu contexto."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    prompt = ChatPromptTemplate.from_template(
        """Você é um avaliador rigoroso. Sua tarefa é verificar a Fact score de uma 'Resposta' fornecida e se as partes dessa resposta podem ser totalmente comprovada usando apenas o 'Contexto' abaixo.
        Analise cada afirmação na 'Resposta'. Compare-a com as informações no 'Contexto'.
        
        Atribua uma nota de 0 a 1 utilizando até duas casas decimais, onde:
        0: Nenhuma parte da resposta é comprovada pelo contexto.
        1: TODAS as partes da resposta são EXPLICITAMENTE comprovadas pelo contexto.
        
        Responda APENAS com o número da nota, sem explicações.

Contexto: 
{context}

Resposta:
{answer}

Avaliação de Fidelidade (0-1):
"""
    )
    return prompt | llm | ScoreParser()



def create_vectorstore(splits, collection_name):
    print("Criando base de vetores")
    model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=model, collection_name=collection_name)
    retriever = vectorstore.as_retriever(kwargs={"K": 7})
    return vectorstore, retriever

def create_rag_chain_with_source(retriever, prompt):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    rag_chain_from_docs = (
        {
            "context": lambda x: format_docs(x["context"]),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source

def set_prompt():
    template = """Você é um assistente honesto, responda utilizando apenas o contexto, se não souber ou não houver no contexto, diga que não sabe.
                  Tente dar a resposta mais detalhada possível
                  Dê os textos que utilizou para usar sua resposta, mencionando o numero do artigo, se houver,
                  por exempo Art. ## inciso ##.
                  também inclua a página para acesso.


                Contexto:
                {context}

                Pergunta:
                {question}


                Resposta: """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt

class Chatbot:
    def __init__(self):
        load_dotenv()
       
        self.url = "https://www.facom.ufms.br/wp-content/uploads/2023/07/20232-EM-DIANTE-ATUAL-RESOLUCAO__COPP__n_704__de_07_07_2023.pdf"
        self.collection_name = "pdf_embeddings"
        splits = load_and_split(self.url)
        _, self.retriever = create_vectorstore(splits, self.collection_name)
        self.prompt = set_prompt()
        self.rag_chain_with_source = create_rag_chain_with_source(self.retriever, self.prompt)
        self.factscore_chain = create_factscore_chain()
        print("Chatbot inicializado e pronto para receber perguntas.")
       
        abs_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(abs_path)
        self.save_path = os.path.join(dir_path, "evaluation")

    async def generate_response(self, question: str):
        if not self.rag_chain_with_source:
            return "O sistema RAG não foi inicializado corretamente. Verifique a configuração."

        result = await self.rag_chain_with_source.ainvoke(question)
        answer = result.get("answer", "Não foi possível gerar uma resposta.")
        context_docs = result.get("context")

        score = 0.0
        if context_docs:
            context_str = format_docs(context_docs)
            score = await self.factscore_chain.ainvoke({"context": context_str, "answer": answer})
            save_score(question, answer, score, self.save_path)
            return f"{answer}\n\n---\n**fact score:** {score:.1f}/1.0" # Formats the output string
        else:
            save_score(question, answer, 0.0, self.save_path)
            return f"{answer}\n\n---\n**fact score:** N/A (no context retrieved)"

    async def process_questions_from_file(self, file_obj):
        """
        Lê perguntas de um arquivo carregado via Gradio, uma por linha, e gera respostas.
        """
        if not file_obj:
            return "Por favor, carregue um arquivo de perguntas (.txt) primeiro."

        filepath = file_obj.name
        print(f"Iniciando processamento em lote do arquivo: {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            return f"Erro: Arquivo não encontrado em {filepath}"

        for i, question in enumerate(questions):
            print(f"--- Processando pergunta {i+1}/{len(questions)}: '{question}' ---")
            response = await self.generate_response(question)
            print(f"Resposta gerada:\n{response}\n")
        
        return f"Processamento em lote concluído para {len(questions)} perguntas. Verifique o console para ver as respostas."

