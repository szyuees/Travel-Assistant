"""
raglite_chatbot.py
------------------
Retrieval-augmented chatbot with automatic LoRA detection.
Loads LoRA adapter if available, otherwise uses base model.
"""

import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from retrieval_faiss import TravelRetriever



# Load model + tokenizer
BASE_MODEL = "google/flan-t5-small"
LORA_DIR = "lora_travel_t5"

print("🔹 Loading model...")

# Always use tokenizer from base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

# If LoRA adapter exists, attach it
if os.path.exists(LORA_DIR) and os.path.exists(os.path.join(LORA_DIR, "adapter_model.safetensors")):
    print(f"✅ LoRA adapter detected in '{LORA_DIR}' — loading fine-tuned weights...")
    model = PeftModel.from_pretrained(model, LORA_DIR)
else:
    print("No LoRA adapter found. Using base model only.")


# Initialize retriever
retriever = TravelRetriever("data/faq_data.json", "data/destinations.json")


# RAG-Lite chatbot
def rag_answer(query):
    # Retrieve top-k relevant contexts
    contexts = retriever.retrieve_city_focused(query, k=3)

    # Combine query and retrieved context into a single prompt
    combined = (
        "You are a helpful travel assistant. Use the context below to answer clearly and factually in 2–3 sentences. "
        "If the question asks what to visit, list attractions.\n\n"
        f"Question: {query}\n\n"
        "Context:\n" + "\n".join(contexts)
    )

    print("\n[Retrieved Context]")
    print("\n".join(contexts))
    print("---------------")

    # Generate answer
    inputs = tokenizer(combined, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=120)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



# Interactive loop
if __name__ == "__main__":
    print("🧭 Travel Assistant Chatbot (RAG-Lite + LoRA Ready)")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("Ask a travel question: ")
        if q.lower() == "exit":
            break
        print("Answer:", rag_answer(q))
        print()