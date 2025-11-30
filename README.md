# Chatbot RAG com Avaliação de Fidelidade

Este projeto foi feito como trabalho para disciplina de Redes Neurais da UFMS em 2025/2 e implementa um chatbot de Recuperação Aumentada de Geração (RAG) que utiliza um documento PDF como base de conhecimento. A interface do usuário é construída com Gradio, permitindo que os usuários façam perguntas e recebam respostas baseadas no conteúdo do PDF, com uma avaliação automática da fidelidade da resposta.

## Funcionalidades

*   **Carregamento de PDF via URL:** O sistema baixa um PDF de uma URL especificada.
*   **Fallback para Arquivo Local:** Caso o download falhe, o sistema utiliza um arquivo PDF local (`data/Novo.pdf`).
*   **Geração de Respostas com RAG:** Utiliza a biblioteca LangChain para dividir o documento, criar embeddings (vetores) e gerar respostas contextuais com um modelo de linguagem do Google (Gemini).
*   **Avaliação de Fidelidade (Faithfulness):** Cada resposta é avaliada em uma escala de 0 a 1 para indicar o quão fiel ela é ao contexto extraído do documento.
*   **Interface Web com Gradio:** Uma interface simples para interagir com o chatbot.
*   **Visualização de Scores:** Gera e exibe um gráfico com os scores de fidelidade das perguntas feitas.
*   **Processamento em Lote:** Permite o upload de um arquivo `.txt` com múltiplas perguntas para serem processadas de uma vez.

## Pré-requisitos

*   Python 3.9 ou superior
*   Uma chave de API do Google para acesso aos modelos Gemini. Você pode obter uma no Google AI Studio.

## Instalação

1.  **Clone o repositório:**
    ```bash
    git clone <URL_DO_SEU_REPOSITORIO>
    cd RAG
    ```

3.  **Instale as dependências:**
    Em um venv ou umam maquina virtual com python

    Execute o comando de instalação:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure sua chave de API:**
    Crie um arquivo chamado `.env` na raiz do projeto e adicione sua chave da API do Google:
    ```
    GOOGLE_API_KEY="SUA_CHAVE_DE_API_AQUI"
    ```

## Como Executar

Para iniciar a aplicação e a interface do Gradio, execute o seguinte comando no terminal, a partir da pasta raiz do projeto:

```bash
python main.py
```

Abra o endereço local (geralmente `http://127.0.0.1:7860`) que aparecerá no seu terminal para acessar a interface do chatbot.