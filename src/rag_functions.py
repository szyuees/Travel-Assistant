# ==========================================
# rag_functions.py
# Core retrieval + generation logic for RAG
# ==========================================

import json, numpy as np, faiss, os
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

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
# 3. Load Gemma-2-2B-IT generator
# ==========================================
def load_generator():
    def load_generator():
    print("🔹 Loading Gemma-2-2B-IT...")
    model_name = "google/gemma-2-2b-it"

    # 可选：4bit 量化（显存不紧可删除 bnb_config 与 quantization_config）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=bnb_config,   # 不用量化就删掉这行
        use_auth_token=True,
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,          # 可按需调整
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return generator, tokenizer

# ==========================================
# 4. Core RAG functions
# ==========================================
def retrieve_context(query, retriever, index, corpus, top_k=3):
    q_emb = retriever.encode([query])
    D, I = index.search(np.array(q_emb, dtype="float32"), k=top_k)
    return [corpus[i] for i in I[0]]

def _pack_context(passages: List[str], max_passages: int = 8) -> str:
    """
    Turn a list of passages into a numbered block:
    [1] passage text...
    [2] passage text...
    """
    lines = []
    for i, p in enumerate(passages[:max_passages], start=1):
        lines.append(f"[{i}] {p.strip()}")
    return "\n\n".join(lines)

def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])

def _truncate_to_budget(tokenizer, text: str, budget_tokens: int) -> str:
    """
    Truncate the text to fit within a token budget using a binary search on characters.
    Keeps things simple and fast while avoiding context overflows.
    """
    if _count_tokens(tokenizer, text) <= budget_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if _count_tokens(tokenizer, text[:mid]) <= budget_tokens:
            lo = mid + 1
        else:
            hi = mid
    return text[:lo-1]

# ---------- Main RAG answering function ----------

def rag_answer(
    query: str,
    retriever,             # SentenceTransformer (or compatible)
    index,                 # FAISS index
    corpus: List[str],     # passages aligned with FAISS ids
    generator,             # HF pipeline("text-generation")
    tokenizer,             # matching tokenizer for Gemma
    top_k: int = 3,
    max_new_tokens: int = 200,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> Tuple[str, List[str]]:
    # 1) Retrieve
    q_emb = retriever.encode([query], convert_to_numpy=True)  # add normalize_embeddings=True if your index was built normalized
    D, I = index.search(np.array(q_emb, dtype="float32"), k=top_k)
    contexts = [corpus[i] for i in I[0]]

    # 2) Build a numbered context block
    context_text = _pack_context(contexts, max_passages=top_k)

    # 3) Input-length budgeting (prevents truncation/overflow)
    #    Try to read model max length; default to 8192 if missing.
    try:
        max_ctx_len = generator.model.config.max_position_embeddings
    except Exception:
        max_ctx_len = 8192
    input_budget = max(256, max_ctx_len - max_new_tokens - 50)  # keep a small safety buffer

    ctx_budget = int(input_budget * 0.8)   # 80% for context
    qry_budget = input_budget - ctx_budget # 20% for the user query

    context_text = _truncate_to_budget(tokenizer, context_text, ctx_budget)
    safe_query = _truncate_to_budget(tokenizer, query, qry_budget)

    # 4) System + User via chat template
    system_prompt = (
        "You are a helpful travel assistant. "
        "Answer ONLY using the provided context. "
        "Cite sources inline like [1], [2] based on the passage indices. "
        "If the answer is not in the context, reply 'I am not sure' briefly."
    )
    messages = [
        {"role": "system", "content": f"{system_prompt}\n\n---\nCONTEXT:\n{context_text}\n---"},
        {"role": "user",   "content": safe_query},
    ]
    templated = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 5) Generate and strip the prompt prefix from pipeline output
    out = generator(
        templated,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )[0]["generated_text"]
    answer = out[len(templated):].strip()
    return answer, contexts

# ==========================================
# 5. Global initialization (so imports work)
# ==========================================
print("🔹 Initializing retriever, index, and corpus...")
retriever, index, corpus = load_faiss_index()
print("✅ rag_functions.py ready.")
