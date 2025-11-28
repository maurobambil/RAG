import llm
import gradio as gr
import asyncio



def main():
    # O Chatbot agora carrega o PDF e inicializa o RAG no construtor.
    chatbot = llm.Chatbot()
    with gr.Blocks() as demo:
        gr.Markdown("Chat RAG com Gradio")

        pergunta_input = gr.Textbox(
            label="Digite sua pergunta:",
            placeholder="Ex: Explique como será a seleção."
        )

        resposta_output = gr.Textbox(
            label="Resposta",
            lines=30
        )

        botao = gr.Button("Enviar")

        # Quando clicar no botão, chama sua função
        botao.click(
            
            fn=chatbot.generate_response,
            inputs=pergunta_input,
            outputs=resposta_output
        )
    # Testa a interface
    demo.launch(share=True)

if __name__ == "__main__":
    # Corrige um problema de loop de eventos do asyncio no Windows
    # que causa o travamento do Gradio com funções assíncronas.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()
        
    

        
    