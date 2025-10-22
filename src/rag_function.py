# ==========================================
# rag_functions.py
# Core retrieval + generation logic for RAG
# ==========================================

import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ==========================================
# 1. Load retriever (BGE-large)
# ==========================================
def load_retriever():
    print("🔹 Loading BGE retriever...")
    return SentenceTransformer("BAAI/bge-large-en-v1.5")

# ==========================================
# 2. Load FAISS index (prebuilt or rebuild if missing)
# ==========================================
def load_faiss_index(
    index_path="data/faiss_index.bin",
    corpus_path="data/corpus_sample.json",
    data_path="data/wikivoyage_chunks.jsonl",
    sample_size=10_000
):
    """
    Loads a saved FAISS index and corresponding corpus.
    If not found, rebuilds using WikiVoyage data.
    """
    import os
    retriever = load_retriever()

    if os.path.exists(index_path) and os.path.exists(corpus_path):
        print("✅ Loading existing FAISS index and corpus...")
        index = faiss.read_index(index_path)
        with open(corpus_path, "r") as f:
            corpus = json.load(f)
    else:
        print("⚙️ Index not found — rebuilding...")
        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f]
        corpus = [x["text"] for x in data[:sample_size]]
        embeddings = retriever.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings, dtype="float32"))
        faiss.write_index(index, index_path)
        with open(corpus_path, "w") as f:
            json.dump(corpus, f)
        print(f"✅ Saved FAISS index to {index_path}")

    return retriever, index, corpus

# ==========================================
# 3. Load Llama-3.2 generator
# ==========================================
def load_generator():
    print("🔹 Loading Llama-3.2 model...")
    model_name = "meta-llama/Llama-3.2-3b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto", use_auth_token=True
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# ==========================================
# 4. Core RAG functions
# ==========================================
def retrieve_context(query, retriever, index, corpus, top_k=3):
    q_emb = retriever.encode([query])
    D, I = index.search(np.array(q_emb, dtype="float32"), k=top_k)
    return [corpus[i] for i in I[0]]

def rag_answer(query, retriever, index, corpus, generator, top_k=3):
    contexts = retrieve_context(query, retriever, index, corpus, top_k)
    context_text = "\n".join(contexts)
    prompt = (
        f"You are a helpful travel assistant. "
        f"Use the information below to answer concisely and factually.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    )
    result = generator(prompt, max_new_tokens=150, temperature=0.6, top_p=0.9)[0]["generated_text"]
    return result.strip(), contexts