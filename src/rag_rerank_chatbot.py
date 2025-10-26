# ==========================================
# rag_rerank_chatbot.py
# Evaluation for RAG with Reranking
# ==========================================
# -*- coding: utf-8 -*-
"""
RAG + Lightweight Rerank + Gemma-2-2B-IT Chatbot

- Retrieve top_k passages via FAISS
- Rerank candidates with embedding cosine similarity (fast, no extra models)
- Pack contexts and apply Gemma chat template
- Generate final answer with Gemma-2-2B-IT

This script expects:
  - load_faiss_index() -> (retriever: SentenceTransformer, index: faiss.Index, corpus: List[str])
  - load_generator()    -> (generator: HF pipeline("text-generation"), tokenizer: AutoTokenizer)
from src.rag_functions.

If your function names differ, adjust the imports accordingly.
"""

from typing import List, Tuple

import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer, util

# Project utilities (adjust paths/names if your module structure differs)
from rag_functions import load_faiss_index, load_generator


# ---------------------------
# Retrieval (top-k from FAISS)
# ---------------------------
def retrieve_topk(
    query: str,
    retriever: SentenceTransformer,
    index,
    corpus: List[str],
    k: int = 10,
) -> Tuple[List[str], List[int], List[float]]:
    """
    Returns (contexts, indices, sims) for the top-k retrieved passages.
    Note: If your FAISS index was built with normalized vectors,
          set normalize_embeddings=True here to match.
    """
    q_emb = retriever.encode([query], convert_to_numpy=True)  # add normalize_embeddings=True if needed
    D, I = index.search(np.array(q_emb, dtype="float32"), k=k)  # D: distance, I: indices
    idxs = I[0].tolist()
    dists = D[0].tolist()
    sims = [-d for d in dists]  # convert L2 distance to a similarity proxy
    contexts = [corpus[i] for i in idxs]
    return contexts, idxs, sims


# ---------------------------
# Lightweight reranker
# ---------------------------
def rerank_contexts(
    query: str,
    candidates: List[str],
    retriever: SentenceTransformer,
    top_rerank: int = 5,
) -> List[str]:
    """
    Rerank candidate passages by cosine similarity using the same retriever.
    Fast and dependency-free alternative to CrossEncoder rerankers.
    """
    if not candidates:
        return []

    q_emb = retriever.encode([query], convert_to_tensor=True, normalize_embeddings=True)
    c_emb = retriever.encode(candidates, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(q_emb, c_emb).cpu().numpy().ravel()  # shape: (n,)
    order = np.argsort(sims)[::-1]
    top_idx = order[:top_rerank]
    return [candidates[i] for i in top_idx]


# ---------------------------
# Packing + token budgeting
# ---------------------------
def pack_context(passages: List[str], max_passages: int = 8) -> str:
    lines = []
    for i, p in enumerate(passages[:max_passages], start=1):
        lines.append(f"[{i}] {p.strip()}")
    return "\n\n".join(lines)


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def truncate_to_budget(tokenizer, text: str, budget_tokens: int) -> str:
    if count_tokens(tokenizer, text) <= budget_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if count_tokens(tokenizer, text[:mid]) <= budget_tokens:
            lo = mid + 1
        else:
            hi = mid
    return text[:lo - 1]


# ---------------------------
# Chat-template generation
# ---------------------------
def answer_with_gemma(
    query: str,
    contexts: List[str],
    generator,
    tokenizer,
    max_new_tokens: int = 300,
    temperature: float = 0.4,
    top_p: float = 0.9,
) -> str:
    """
    Build a chat-formatted prompt for Gemma and generate the final answer.
    """
    context_block = pack_context(contexts, max_passages=8)

    # Read model context length; default conservatively if absent.
    try:
        max_ctx = generator.model.config.max_position_embeddings
    except Exception:
        max_ctx = 8192

    # Reserve some budget for generation and a small safety buffer.
    input_budget = max(256, max_ctx - max_new_tokens - 50)
    ctx_budget = int(input_budget * 0.8)
    qry_budget = input_budget - ctx_budget

    context_block = truncate_to_budget(tokenizer, context_block, ctx_budget)
    safe_query = truncate_to_budget(tokenizer, query, qry_budget)

    system_prompt = (
        "You are a travel assistant. Use ONLY the provided context to answer. "
        "Cite sources inline like [1], [2] based on the passage indices. "
        "If the answer is not in the context, reply 'I am not sure' briefly."
    )
    messages = [
        {"role": "system", "content": f"{system_prompt}\n\n---\nCONTEXT:\n{context_block}\n---"},
        {"role": "user", "content": safe_query},
    ]
    templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    out = generator(
        templated,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )[0]["generated_text"]

    return out[len(templated):].strip()


# ---------------------------
# RAG → Rerank → Answer
# ---------------------------
def rag_rerank_answer(
    query: str,
    retriever: SentenceTransformer,
    index,
    corpus: List[str],
    generator,
    tokenizer,
    top_k: int = 10,
    top_rerank: int = 5,
) -> Tuple[str, List[str]]:
    candidates, _, _ = retrieve_topk(query, retriever, index, corpus, k=top_k)
    ranked = rerank_contexts(query, candidates, retriever, top_rerank=top_rerank)
    answer = answer_with_gemma(query, ranked, generator, tokenizer, max_new_tokens=300, temperature=0.4, top_p=0.9)
    return answer, ranked


# ---------------------------
# Gradio app
# ---------------------------
def build_app(retriever, index, corpus, generator, tokenizer):
    def chat_response(q: str):
        if not q.strip():
            return "Please enter a question.", ""
        ans, ctx = rag_rerank_answer(
            q, retriever, index, corpus, generator, tokenizer,
            top_k=10, top_rerank=5
        )
        ctx_preview = "\n\n---\n\n".join(ctx[:3])
        return f"**Answer:**\n{ans}", f"**Top Contexts (after rerank):**\n{ctx_preview}"

    return gr.Interface(
        fn=chat_response,
        inputs=gr.Textbox(label="Ask a travel question", placeholder="e.g., How to get from Changi Airport to Marina Bay Sands?"),
        outputs=[gr.Markdown(label="Response"), gr.Markdown(label="Contexts")],
        title="Travel Assistant — RAG + Rerank (Gemma-2-2B-IT)",
        description="Retrieval with lightweight rerank, then answer with Gemma-2-2B-IT using chat templates.",
        theme="soft",
    )


# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    # 1) Index + corpus
    retriever, index, corpus = load_faiss_index()  # must return (retriever, index, corpus)

    # 2) Generator + tokenizer (Gemma-2-2B-IT)
    generator, tokenizer = load_generator()        # must return (generator, tokenizer)

    # 3) Launch Gradio
    app = build_app(retriever, index, corpus, generator, tokenizer)
    app.launch(share=False)
