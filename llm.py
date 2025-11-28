from utils import load_pdf, split_pdf, format_docs
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
#from langchain.schema import BaseOutputParser
import re


class ScoreParser(BaseOutputParser):
    def parse(self, text: str) -> float:
        try:
            return float(text.strip())
        except ValueError:
            match = re.search(r'\d+(\.\d+)?', text)
            if match:
                return float(match.group(0))
            return 0.0

def create_faithfulness_chain():
    """Cria uma cadeia para pontuar a fidelidade de uma resposta ao seu contexto."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    prompt = ChatPromptTemplate.from_template(
        """Você é um avaliador rigoroso. Sua tarefa é verificar a fidelidade de uma 'Resposta' fornecida e se ela pode ser totalmente comprovada usando apenas o 'Contexto' abaixo.

        Analise cada afirmação na 'Resposta'. Compare-a com as informações no 'Contexto'.
        
        Atribua uma nota de 1 a 5, onde:
        1: Nenhuma parte da resposta é comprovada pelo contexto.
        5: TODAS as partes da resposta são EXPLICITAMENTE comprovadas pelo contexto.
        
        Responda APENAS com o número da nota, sem explicações.

Contexto: 
{context}

Resposta:
{answer}

Avaliação de Fidelidade (1-5):
"""
    )
    return prompt | llm | ScoreParser()


def load_and_split(url):
    print(f"Carregando o PDF da URL: {url}...")
    tmp_path = load_pdf(url)
    print("Dividindo o PDF em partes...")
    splits = split_pdf(tmp_path)
    return splits

def create_vectorstore(splits, collection_name):
    print("Criando base de vetores")
    model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=model, collection_name=collection_name)
    retriever = vectorstore.as_retriever(kwargs={"K": 7})
    return vectorstore, retriever

def create_rag_chain_with_source(retriever, prompt):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

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
        self.faithfulness_chain = create_faithfulness_chain()
        print("Chatbot inicializado e pronto para receber perguntas.")

    async def generate_response(self, question: str):
        if not self.rag_chain_with_source:
            return "O sistema RAG não foi inicializado corretamente. Verifique a configuração."

        result = await self.rag_chain_with_source.ainvoke(question)
        answer = result.get("answer", "Não foi possível gerar uma resposta.")
        context_docs = result.get("context")

        score = 0.0
        if context_docs:
            context_str = format_docs(context_docs)
            score = await self.faithfulness_chain.ainvoke({"context": context_str, "answer": answer})
            return f"{answer}\n\n---\n**Faithfulness:** {score:.1f}/5" # Formats the output string
        else:
            return f"{answer}\n\n---\n**Faithfulness:** N/A (no context retrieved)"


def calc_distances(question, vectorstore):
    pass
