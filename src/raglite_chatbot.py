# ==========================================
# raglite_chatbot.py
# Evaluation pipeline for RAG-Lite (Top-k)
# ==========================================

from rag_functions import (
    load_retriever,
    load_faiss_index,
    load_generator,
    rag_answer,
)
import pandas as pd
from sentence_transformers import util
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("🔹 Initializing components...")
    retriever, index, corpus = load_faiss_index()
    generator = load_generator()

    # ======================
    # Run evaluation
    # ======================
    print("🔹 Evaluating RAG-Lite (Top-k=3)...")
    df = pd.read_csv("data/evaluation_queries.csv")
    results = []

    for _, row in df.iterrows():
        q, ref = row["query"], row["expected_answer"]
        ans, _ = rag_answer(q, retriever, index, corpus, generator, top_k=3)
        sim = util.cos_sim(
            retriever.encode(ans, convert_to_tensor=True),
            retriever.encode(ref, convert_to_tensor=True)
        ).item()
        results.append({"query": q, "answer": ans, "similarity": sim})

    df_out = pd.DataFrame(results)
    avg_sim = df_out["similarity"].mean()
    print(f"✅ Average similarity: {avg_sim:.4f}")

    # ======================
    # Save + visualize results
    # ======================
    output_path = "reports/evaluation_llama3.2_RAG.csv"
    df_out.to_csv(output_path, index=False)
    print(f"💾 Saved evaluation results to {output_path}")

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(df_out)), df_out["similarity"], color="teal")
    plt.title(
        f"RAG Similarity per Query (Llama-3.2 + BGE + WikiVoyage)\nAvg Similarity: {avg_sim:.3f}"
    )
    plt.xlabel("Query Index")
    plt.ylabel("Cosine Similarity")
    plt.tight_layout()
    plt.show()