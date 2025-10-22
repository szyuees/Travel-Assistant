# ==========================================
# rag_rerank_chatbot.py
# Evaluation for RAG with Reranking
# ==========================================
from rag_functions import rag_answer, retriever, corpus
from sentence_transformers import CrossEncoder, util
import pandas as pd, numpy as np, matplotlib.pyplot as plt

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_and_rerank(query, top_k=10, rerank_top=3):
    q_emb = retriever.encode([query])
    D, I = retriever.encode([query]), None
    D, I = index.search(np.array(q_emb, dtype="float32"), k=top_k)
    candidates = [corpus[i] for i in I[0]]

    pairs = [(query, passage) for passage in candidates]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [text for text, _ in reranked[:rerank_top]]

def rag_reranked_answer(query):
    contexts = retrieve_and_rerank(query)
    context_text = "\n".join(contexts)
    prompt = (
        f"You are a helpful travel assistant. Use the information below to answer clearly and factually.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    )
    result = generator(prompt)[0]["generated_text"]
    return result.strip(), contexts

if __name__ == "__main__":
    df = pd.read_csv("data/evaluation_queries.csv")
    results = []

    for _, row in df.iterrows():
        q, ref = row["query"], row["expected_answer"]
        ans, _ = rag_reranked_answer(q)
        sim = util.cos_sim(
            retriever.encode(ans, convert_to_tensor=True),
            retriever.encode(ref, convert_to_tensor=True)
        ).item()
        results.append({"query": q, "answer": ans, "similarity": sim})

    df_rerank = pd.DataFrame(results)
    print("Average similarity (Reranked RAG):", df_rerank["similarity"].mean())
    df_rerank.to_csv("reports/evaluation_llama3.2_RAG_reranked.csv", index=False)