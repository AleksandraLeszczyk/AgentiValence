import gradio as gr
from dotenv import load_dotenv

from principal_investigator import chat

load_dotenv(override=True)


def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Quantum Chemistry Lab", theme=theme) as ui:
        gr.Markdown("# 🏢 QC Expert Assistant\nAsk me anything about Quantum Chemistry!")

        with gr.Row():
            with gr.Column(scale=1):
                # CRITICAL: Add type="messages" to support the dict format
                chatbot = gr.Chatbot(label="💬 Lab") 
                message = gr.Textbox(
                    label="Your Research Project",
                    placeholder="Ask anything about quantum chemistry...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Documents",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        message.submit(
            put_message_in_chatbot,
            inputs=[message, chatbot],
            outputs=[message, chatbot],
        ).then(
            chat, 
            inputs=[chatbot], 
            outputs=[chatbot, context_markdown]
        )

    ui.launch(inbrowser=True)

if __name__ == "__main__":
    main()
