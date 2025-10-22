# ==========================================
# gradio_app.py
# Interactive demo for Travel Assistant Chatbot (RAG-Lite)
# ==========================================

import gradio as gr
from rag_functions import load_retriever, load_faiss_index, load_generator, rag_answer

# ==========================================
# Load models once (cached globally)
# ==========================================
print("🔹 Initializing components for Gradio app...")
retriever, index, corpus = load_faiss_index()
generator = load_generator()

# ==========================================
# Chat logic
# ==========================================
def chat_response(query):
    answer, contexts = rag_answer(query, retriever, index, corpus, generator, top_k=3)
    context_preview = "\n\n---\n\n".join(contexts[:2])
    return f"**Answer:** {answer}\n\n**Top Retrieved Contexts:**\n{context_preview}"

# ==========================================
# Gradio Interface
# ==========================================
iface = gr.Interface(
    fn=chat_response,
    inputs=gr.Textbox(
        label="Ask a travel question:",
        placeholder="e.g. How do I get from Changi Airport to Marina Bay Sands?",
    ),
    outputs=gr.Markdown(label="Response"),
    title="🌍 Travel Assistant Chatbot (RAG-Lite)",
    description="Ask any travel-related question. This chatbot uses WikiVoyage data and Llama-3.2-3B-Instruct for factual travel answers.",
    theme="soft",
)

if __name__ == "__main__":
    iface.launch(share=True)