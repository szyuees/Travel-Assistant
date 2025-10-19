"""
evaluation.py
--------------
Compare Baseline, RAG-Lite, and LoRA-tuned models
using semantic similarity on evaluation_queries.csv.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from retrieval_faiss import TravelRetriever


# 1. Load evaluation data
df = pd.read_csv("data/evaluation_queries.csv")


# 2. Load models
print("Loading models...")

BASE_MODEL = "google/flan-t5-small"
LORA_DIR = "lora_travel_t5"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model_base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
model_rag = model_base

# Load LoRA adapter if available
if True:  # set to False if you want to skip LoRA
    model_lora = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model_lora = PeftModel.from_pretrained(model_lora, LORA_DIR)
else:
    model_lora = None

retriever = TravelRetriever("data/faq_data.json", "data/destinations.json")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Helper functions
def gen_base_answer(query):
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model_base.generate(**inputs, max_new_tokens=80)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def gen_rag_answer(query):
    contexts = retriever.retrieve_city_focused(query, k=3)
    combined = (
        "You are a helpful travel assistant. Read the context and answer in 2–3 sentences.\n\n"
        f"Question: {query}\n\nContext:\n" + "\n".join(contexts)
    )
    inputs = tokenizer(combined, return_tensors="pt", truncation=True)
    outputs = model_rag.generate(**inputs, max_new_tokens=120)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def gen_lora_answer(query):
    contexts = retriever.retrieve_city_focused(query, k=3)
    combined = (
        "You are a travel assistant fine-tuned on travel FAQs. Answer concisely using the context below.\n\n"
        f"Question: {query}\n\nContext:\n" + "\n".join(contexts)
    )
    inputs = tokenizer(combined, return_tensors="pt", truncation=True)
    outputs = model_lora.generate(**inputs, max_new_tokens=120)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def similarity(a, b):
    ea, eb = encoder.encode(a, convert_to_tensor=True), encoder.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(ea, eb))

# 4. Evaluate all models
results = []
print("🔹 Running evaluation (Baseline, RAG, LoRA)...")

for i, row in df.iterrows():
    q, ref = row["query"], row["expected_answer"]
    base = gen_base_answer(q)
    rag = gen_rag_answer(q)
    lora = gen_lora_answer(q)

    results.append({
        "Query": q,
        "Baseline": base,
        "RAG": rag,
        "LoRA": lora,
        "Sim_Baseline": similarity(base, ref),
        "Sim_RAG": similarity(rag, ref),
        "Sim_LoRA": similarity(lora, ref)
    })

df_out = pd.DataFrame(results)
df_out.to_csv("reports/evaluation_lora_results.csv", index=False)

print("Evaluation complete.")
print(df_out[["Query", "Sim_Baseline", "Sim_RAG", "Sim_LoRA"]].head())

print("\nAverage similarity:")
print(df_out[["Sim_Baseline", "Sim_RAG", "Sim_LoRA"]].mean())