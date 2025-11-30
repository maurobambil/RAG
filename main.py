#Authors: Mauro Bambil de Paula e Juan Carlos Conceição de Lima Sales
import llm
import gradio as gr
import asyncio
from plot_scores import plot_faithfulness_scores
import os

def show_plot():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(dir_path, 'evaluation', 'faithfulness_scores.csv')
    plot_filename = os.path.join(dir_path, 'evaluation', 'faithfulness_scores_plot.png')
    if not os.path.exists(csv_path):
        return None
    plot_faithfulness_scores(csv_path, save_path=os.path.join(dir_path,'evaluation'))
    return plot_filename

def main():
    # O Chatbot agora carrega o PDF e inicializa o RAG no construtor.
    chatbot = llm.Chatbot()
    with gr.Blocks() as demo:
        gr.Markdown("Chat RAG com Fact Score")
        
        with gr.Row():
            with gr.Column():
                pergunta_input = gr.Textbox(
                    label="Digite sua pergunta:",
                    placeholder="Ex: Explique como será a seleção."
                )

                resposta_output = gr.Textbox(
                    label="Resposta",
                    lines=30
                )

                enviar_botao = gr.Button("Enviar")

            with gr.Column():
                gr.Markdown("### Análise de Fact Score")
                plot_botao = gr.Button("Atualizar Gráfico de Fact Score")
                plot_output = gr.Image(label="Fact Score Plot")

        with gr.Row():
            with gr.Column():
                gr.Markdown("---")
                gr.Markdown("### Processamento em Lote")
                gr.Markdown("Carregue um arquivo .txt com uma pergunta por linha para processar várias de uma vez.\n\n" \
                "Este processo demorará alguns minutos dependendo do tamanho do arquivo.")
                
                arquivo_input = gr.File(label="Arquivo de Perguntas (.txt)")
                
                processar_arquivo_botao = gr.Button("Processar Arquivo")

                status_processamento_output = gr.Textbox(label="Status do Processamento", interactive=False)

        # Quando clicar no botão, chama sua função
        enviar_botao.click(
            fn=chatbot.generate_response,
            inputs=pergunta_input,
            outputs=resposta_output
        )

        processar_arquivo_botao.click(
            fn=chatbot.process_questions_from_file,
            inputs=arquivo_input,
            outputs=status_processamento_output
        )

        plot_botao.click(
            fn=show_plot,
            inputs=None,
            outputs=plot_output
        )
    # Testa a interface
    demo.launch(share=True)

if __name__ == "__main__":

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()