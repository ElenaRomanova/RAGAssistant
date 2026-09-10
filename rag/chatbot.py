import json
import gradio as gr
import rag.rag_traced


def dummy_rag_pipeline(message: str, history):
    print(f"Received message: {message}")
    answer = rag.rag_traced.answer(message)
    print(answer)
    bot_response = f"Ответ: '{answer.get('answer')}'."
    if len(answer.get('sources')) > 0:
        bot_response += f"\nДокументы: {answer.get('sources')}"
    return bot_response


demo = gr.ChatInterface(
    fn=dummy_rag_pipeline,
    title="RAG Assistant",
    description="Ask me about the Regulatory Legal Acts of the Republic of Kazakhstan",
    textbox=gr.Textbox(placeholder="Your question...", container=False, scale=7, submit_btn="Answer"),
)

if __name__ == "__main__":
    demo.launch()