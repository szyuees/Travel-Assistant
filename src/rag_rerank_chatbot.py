# RAG with reranker
# ============================================
# Reranker (improves context quality)
# ============================================
from sentence_transformers import CrossEncoder

# Load a compact reranker fine-tuned for MS MARCO relevance tasks
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_and_rerank(query, top_k=10, rerank_top=3):
    """
    1. Retrieve top_k passages via FAISS
    2. Rerank them using a cross-encoder
    3. Return top rerank_top passages
    """
    # --- Stage 1: FAISS dense retrieval ---
    q_emb = retriever.encode([query])
    D, I = index.search(np.array(q_emb, dtype="float32"), k=top_k)
    candidates = [corpus[i] for i in I[0]]

    # --- Stage 2: Reranking with CrossEncoder ---
    pairs = [(query, passage) for passage in candidates]
    scores = reranker.predict(pairs)

    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top_contexts = [text for text, _ in reranked[:rerank_top]]
    return top_contexts

def rag_reranked_answer(query):
    contexts = retrieve_and_rerank(query, top_k=10, rerank_top=3)
    context_text = "\n".join(contexts)
    prompt = (
        f"You are a helpful travel assistant. Use the information below to answer clearly and factually.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    )
    result = generator(prompt)[0]["generated_text"]
    return result.strip(), contexts

results_reranked = []
for _, row in df.iterrows():
    q, ref = row["query"], row["expected_answer"]
    ans, _ = rag_reranked_answer(q)
    sim = util.cos_sim(
        retriever.encode(ans, convert_to_tensor=True),
        retriever.encode(ref, convert_to_tensor=True)
    ).item()
    results_reranked.append({"query": q, "answer": ans, "similarity": sim})

df_rerank = pd.DataFrame(results_reranked)
print("Average similarity (Reranked RAG):", df_rerank["similarity"].mean())

df_rerank.to_csv("reports/evaluation_llama3.2_RAG_reranked.csv", index=False)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
plt.bar(["Top-k RAG", "Reranked RAG"],
        [df_out["similarity"].mean(), df_rerank["similarity"].mean()],
        color=["skyblue", "lightgreen"])
plt.title("Average Semantic Similarity Comparison")
plt.ylabel("Cosine Similarity")
plt.show()




