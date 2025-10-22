

# Using RAG-Lite(top-k,k=3)
def retrieve_context(query, top_k=3):
    q_emb = retriever.encode([query])
    D, I = index.search(np.array(q_emb, dtype="float32"), k=top_k)
    return [corpus[i] for i in I[0]]

def rag_answer(query):
    contexts = retrieve_context(query)
    context_text = "\n".join(contexts)
    prompt = (
        f"You are a helpful travel assistant. Use the information below to answer factually and concisely.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    )
    result = generator(prompt)[0]["generated_text"]
    return result.strip(), contexts

# evaluate with evaluation_queries.csv
import pandas as pd
from sentence_transformers import util

df = pd.read_csv("data/evaluation_queries.csv")
results = []

for _, row in df.iterrows():
    q, ref = row["query"], row["expected_answer"]
    ans, _ = rag_answer(q)
    sim = util.cos_sim(
        retriever.encode(ans, convert_to_tensor=True),
        retriever.encode(ref, convert_to_tensor=True)
    ).item()
    results.append({"query": q, "answer": ans, "similarity": sim})

df_out = pd.DataFrame(results)
print("Average similarity:", df_out["similarity"].mean())

df_out.to_csv("reports/evaluation_llama3.2_RAG.csv", index=False)


# visualise the improvement
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
plt.bar(range(len(df_out)), df_out["similarity"])
plt.title("RAG Similarity per Query (Llama-3.2 + BGE + WikiVoyage)")
plt.xlabel("Query Index")
plt.ylabel("Cosine Similarity")
plt.show()


