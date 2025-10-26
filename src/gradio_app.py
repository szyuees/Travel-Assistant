# ==========================================
# gradio_app.py
# Interactive demo for Travel Assistant Chatbot (RAG-Lite)
# ==========================================

# src/gradio_app.py
import gradio as gr
from rag_functions import load_faiss_index, load_generator, rag_answer

retriever, index, corpus = load_faiss_index()
generator, tokenizer = load_generator()

def chat_response(query: str):
    ans, ctx = rag_answer(query, retriever, index, corpus, generator, tokenizer, top_k=3)
    preview = "\n\n---\n\n".join(ctx[:2])
    return f"**Answer**\n{ans}\n\n**Top Contexts**\n{preview}"

iface = gr.Interface(
    fn=chat_response,
    inputs=gr.Textbox(label="Ask a travel question", placeholder="e.g., 3-day Singapore food itinerary?"),
    outputs=gr.Markdown(label="Response"),
    title="Travel Assistant (Gemma-2-2B-IT, RAG-Lite)",
    description="Retrieval-augmented answers using Gemma-2-2B-IT with chat templates.",
)

if __name__ == "__main__":
    iface.launch(share=False)
