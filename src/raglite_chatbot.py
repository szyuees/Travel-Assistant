from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from retrieval_faiss import TravelRetriever

# =========================
# Load model + tokenizer
# =========================
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# =========================
# Initialize retriever (FAQ + Destinations)
# =========================
retriever = TravelRetriever("data/faq_data.json", "data/destinations.json")

# =========================
# RAG-Lite chatbot
# =========================
def rag_answer(query):
    # Retrieve top contexts
    contexts = retriever.retrieve(query, k=3)

    # Combine context + query into a single prompt
    combined = (
        "You are a travel assistant. Use the relevant info below to answer clearly.\n\n"
        f"Question: {query}\n\n"
        "Relevant Info:\n" + "\n".join(contexts)
    )

    print("\n[Retrieved Context]")
    print("\n".join(contexts))
    print("---------------")

    # Generate answer
    inputs = tokenizer(combined, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=120)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =========================
# Run chatbot interactively
# =========================
if __name__ == "__main__":
    while True:
        q = input("Ask a travel question (or 'exit' to quit): ")
        if q.lower() == "exit":
            break
        print("Answer:", rag_answer(q))
        print()