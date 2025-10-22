# ==========================================
# retrieval_faiss.py
# Builds and saves FAISS index for WikiVoyage corpus
# ==========================================

import json, random, numpy as np, faiss
from sentence_transformers import SentenceTransformer

def build_faiss_index(
    data_path="data/wikivoyage_chunks.jsonl",
    sample_size=10_000,
    save_path="data/faiss_index.bin"
):
    """
    Builds FAISS index using BGE embeddings.
    Returns the retriever, corpus, and FAISS index.
    """
    print("🔹 Loading data from:", data_path)
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    print("🔹 Sampling corpus of", sample_size, "chunks...")
    sampled_data = random.sample(data, sample_size)
    corpus = [x["text"] for x in sampled_data]

    print("🔹 Encoding with BGE-Large...")
    retriever = SentenceTransformer("BAAI/bge-large-en-v1.5")
    embeddings = retriever.encode(corpus, convert_to_numpy=True, show_progress_bar=True)

    print("🔹 Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype="float32"))
    print(f"✅ Indexed {index.ntotal} passages.")

    # Optional: Save index + corpus for reuse
    faiss.write_index(index, save_path)
    with open("data/corpus_sample.json", "w") as f:
        json.dump(corpus, f)
    print(f"💾 Saved index to {save_path} and corpus_sample.json")

    return retriever, index, corpus


if __name__ == "__main__":
    build_faiss_index()