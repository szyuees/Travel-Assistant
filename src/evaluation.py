"""
evaluation.py
--------------
Compare baseline Flan-T5 vs RAG-Lite chatbot
using semantic similarity on evaluation_queries.csv.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from retrieval_faiss import TravelRetriever

# ======================================================
# 1. Load evaluation data
# ======================================================
df = pd.read_csv("data/evaluation_queries.csv")

# ======================================================
# 2. Load models
# ======================================================
print("🔹 Loading models...")

# Baseline model (prompt-only)
base_name = "google/flan-t5-small"
tokenizer_base = AutoTokenizer.from_pretrained(base_name)
model_base = AutoModelForSeq2SeqLM.from_pretrained(base_name)

# RAG-Lite model
tokenizer_rag = tokenizer_base
model_rag = model_base
retriever = TravelRetriever("data/faq_data.json", "data/destinations.json")

# Semantic similarity encoder
encoder = SentenceTransformer("all-MiniLM-L6-v2")


# ======================================================
# 3. Helper functions
# ======================================================
def generate_baseline_answer(query):
    """Simple prompt-only generation."""
    inputs = tokenizer_base(query, return_tensors="pt")
    outputs = model_base.generate(**inputs, max_new_tokens=80)
    return tokenizer_base.decode(outputs[0], skip_special_tokens=True)


def generate_rag_answer(query):
    """Retrieval-augmented generation."""
    contexts = retriever.retrieve_city_focused(query, k=3)
    combined = (
        "You are a helpful travel assistant. Read the context and answer clearly in 2–3 sentences.\n\n"
        f"Question: {query}\n\nContext:\n" + "\n".join(contexts)
    )
    inputs = tokenizer_rag(combined, return_tensors="pt", truncation=True)
    outputs = model_rag.generate(**inputs, max_new_tokens=120)
    return tokenizer_rag.decode(outputs[0], skip_special_tokens=True)


def semantic_similarity(a, b):
    """Compute cosine similarity between two strings."""
    emb_a = encoder.encode(a, convert_to_tensor=True)
    emb_b = encoder.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(emb_a, emb_b))


# ======================================================
# 4. Run evaluation
# ======================================================
results = []
print("🔹 Running evaluation... (this may take a few minutes)")

for i, row in df.iterrows():
    query = row["query"]
    reference = row["expected_answer"]

    base_ans = generate_baseline_answer(query)
    rag_ans = generate_rag_answer(query)

    sim_base = semantic_similarity(base_ans, reference)
    sim_rag = semantic_similarity(rag_ans, reference)

    results.append({
        "Query": query,
        "Reference": reference,
        "Baseline_Answer": base_ans,
        "RAG_Answer": rag_ans,
        "Sim_Baseline": round(sim_base, 3),
        "Sim_RAG": round(sim_rag, 3)
    })

results_df = pd.DataFrame(results)
results_df.to_csv("reports/evaluation_results.csv", index=False)

# ======================================================
# 5. Print summary
# ======================================================
mean_base = results_df["Sim_Baseline"].mean()
mean_rag = results_df["Sim_RAG"].mean()

print("\n✅ Evaluation complete!")
print(results_df[["Query", "Sim_Baseline", "Sim_RAG"]])
print("\nAverage similarity:")
print(f" - Baseline : {mean_base:.3f}")
print(f" - RAG-Lite : {mean_rag:.3f}")

if mean_rag > mean_base:
    print("\n🎯 RAG-Lite shows improvement in factual grounding and relevance.")
else:
    print("\n⚠️ RAG-Lite did not outperform baseline; review retrieval or prompt design.")