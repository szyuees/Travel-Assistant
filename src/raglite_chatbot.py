# src/raglite_chatbot.py  (REPLACE)
"""
Robust RAG-Lite chatbot that:
- Handles HF download timeouts with retries / local fallback
- Truncates contexts per-token to avoid losing evidence to tokenizer truncation
- Safely loads and uses LoRA adapter if present
- Avoids formatting crashes (defensive final_score handling)
"""

import os
import time
import math
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from retrieval_faiss import TravelRetriever

BASE_MODEL = "google/flan-t5-small"
LORA_DIR = "lora_output/checkpoint-12"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# generation defaults (same as baseline to compare fairly)
GEN_KWARGS = {
    "num_beams": 4,
    "max_new_tokens": 120,
    "do_sample": False,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
}

# Prompt template that prefers context but is not overly strict (keeps factuality)
PROMPT_TEMPLATE = (
    "You are a helpful travel assistant. Use the information in CONTEXTS to answer the question "
    "as concisely as possible (2-3 sentences). If the information is not present in the contexts, "
    "say 'I don't know'. Show the most relevant context id if applicable.\n\nCONTEXTS:\n{contexts}\n\nQUESTION: {question}\nANSWER:"
)


# --- robust HF loader with retries + local fallback ---
def hf_from_pretrained_with_retries(cls, model_name, max_retries=4, sleep_base=1, **kwargs):
    """
    Try to load a model/tokenizer with exponential backoff. On final attempt, try with local_files_only=True
    to use cached files (if available).
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            # do not trust remote code by default (safer)
            return cls.from_pretrained(model_name, trust_remote_code=False, **kwargs)
        except Exception as e:
            last_exc = e
            wait = sleep_base * (2 ** attempt)
            print(f"Warning: attempt {attempt+1} failed for {model_name}: {e}")
            if attempt < max_retries:
                print(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                # final fallback: try local_files_only
                try:
                    print("Final attempt: trying local_files_only=True (use cached files if available)")
                    return cls.from_pretrained(model_name, local_files_only=True, trust_remote_code=False, **kwargs)
                except Exception as e2:
                    print("Local fallback failed:", e2)
                    raise last_exc


# Load tokenizer and base model robustly
print("🔹 Loading tokenizer and base model (with retries)...")
tokenizer = hf_from_pretrained_with_retries(AutoTokenizer, BASE_MODEL)
base_model = hf_from_pretrained_with_retries(AutoModelForSeq2SeqLM, BASE_MODEL)
# Move model to device only after possibly attaching adapter
base_model.to(DEVICE)


# Attach LoRA adapter if present
def load_model_with_optional_peft(base_model, adapter_dir):
    if os.path.isdir(adapter_dir) and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        try:
            print(f"✅ LoRA adapter detected in '{adapter_dir}' — loading adapter.")
            model_peft = PeftModel.from_pretrained(base_model, adapter_dir)
            return model_peft
        except Exception as e:
            print("❗ Failed to load LoRA adapter; falling back to base model. Error:", e)
            return base_model
    else:
        print("No LoRA adapter found: using base model only.")
        return base_model


model = load_model_with_optional_peft(base_model, LORA_DIR)
model.to(DEVICE)
model.eval()


# Initialize retriever (assumes your retrieval_faiss saves index & meta to disk)
retriever = TravelRetriever("data/faq_data.json", "data/destinations.json", verbose=False)


# --- utilities to truncate contexts safely using tokenizer ---
def truncate_text_to_tokens(text, max_tokens, tokenizer):
    """Return text truncated to at most max_tokens tokens (approx exact via tokenizer)."""
    enc = tokenizer(text, truncation=True, max_length=max_tokens, add_special_tokens=False)
    return tokenizer.decode(enc["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)


def build_prompt_with_token_budget(question, contexts, tokenizer, per_context_tokens=120, reserved_for_question=64):
    """
    Build a prompt ensuring the encoder input does not exceed tokenizer.model_max_length.
    - contexts: list of dicts from retriever
    - per_context_tokens: how many tokens to keep per context snippet
    - reserved_for_question: token budget reserved for question+template
    Returns (prompt_text, truncated_flag)
    """
    model_max = getattr(tokenizer, "model_max_length", 512)
    # ensure sanity
    model_max = min(int(model_max), 1024)

    # truncate each context to per_context_tokens
    snippets = []
    for i, c in enumerate(contexts, start=1):
        text = c.get("text", "")
        truncated = truncate_text_to_tokens(text, per_context_tokens, tokenizer)
        meta = c.get("meta", {})
        src = meta.get("source", "unknown")
        sid = meta.get("id", "")
        city = meta.get("city", "")
        # display score (fallback chain)
        display_score = float(c.get("final_score") or c.get("combined_score") or c.get("score") or 0.0)
        header = f"[{i} | {src}#{sid}{(' | ' + city) if city else ''} | score={display_score:.3f}]"
        snippets.append(f"{header}\n{truncated}")

    # assemble contexts block
    contexts_block = "\n\n".join(snippets)

    prompt = PROMPT_TEMPLATE.format(contexts=contexts_block, question=question)

    # Now ensure tokenized prompt will fit into model_max; if not, iteratively reduce per_context_tokens
    enc = tokenizer(prompt, truncation=False, add_special_tokens=True)
    token_len = len(enc["input_ids"])
    truncated_flag = False
    if token_len > model_max:
        truncated_flag = True
        # compute how many tokens to allocate per context roughly
        num_ctx = max(1, len(contexts))
        # compute available for contexts
        available_for_contexts = max(0, model_max - reserved_for_question)
        new_per_ctx = max(16, available_for_contexts // num_ctx)
        # rebuild snippets at smaller per_context
        snippets = []
        for i, c in enumerate(contexts, start=1):
            text = c.get("text", "")
            truncated = truncate_text_to_tokens(text, new_per_ctx, tokenizer)
            meta = c.get("meta", {})
            src = meta.get("source", "unknown")
            sid = meta.get("id", "")
            city = meta.get("city", "")
            display_score = float(c.get("final_score") or c.get("combined_score") or c.get("score") or 0.0)
            header = f"[{i} | {src}#{sid}{(' | ' + city) if city else ''} | score={display_score:.3f}]"
            snippets.append(f"{header}\n{truncated}")
        contexts_block = "\n\n".join(snippets)
        prompt = PROMPT_TEMPLATE.format(contexts=contexts_block, question=question)
        # final check (tokenizer will truncate if still too long)
        enc2 = tokenizer(prompt, truncation=True, max_length=model_max, add_special_tokens=True)
        # decode back to text representation (final prompt passed to model)
        prompt = tokenizer.decode(enc2["input_ids"], skip_special_tokens=True)
    return prompt, truncated_flag


def format_contexts_for_display(items):
    """Return list of short display tuples for UI printing: (source, city, display_score, text_preview)."""
    out = []
    for c in items:
        meta = c.get("meta", {})
        src = meta.get("source", "unknown")
        city = meta.get("city", "")
        display_score = float(c.get("final_score") or c.get("combined_score") or c.get("score") or 0.0)
        text_preview = c.get("text", "").strip()
        out.append((src, city, display_score, text_preview))
    return out


def rag_answer(query, top_k=3, per_context_tokens=120):
    # Retrieve structured contexts (city-focused)
    contexts = retriever.retrieve_city_focused(query, k=top_k)
    if not contexts:
        contexts = retriever.retrieve(query, k=top_k)

    # Build a safe prompt ensuring we don't lose contexts to truncation
    prompt, truncated = build_prompt_with_token_budget(
        question=query, contexts=contexts, tokenizer=tokenizer, per_context_tokens=per_context_tokens
    )

    if truncated:
        print("⚠️ Prompt truncated to fit model max length — contexts were shortened. Consider lowering per_context_tokens.")

    # Tokenize & move to device (ensure max_length uses tokenizer.model_max_length)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=min(1024, tokenizer.model_max_length))
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, **GEN_KWARGS)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer, contexts


# CLI
if __name__ == "__main__":
    print("🧭 Travel Assistant Chatbot (RAG-Lite + LoRA Ready)")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("Ask a travel question: ")
        if q.strip().lower() == "exit":
            break
        ans, ctxs = rag_answer(q, top_k=3, per_context_tokens=120)
        print("\n[Answer]\n", ans)
        print("\n[Top contexts used]")
        display_list = format_contexts_for_display(ctxs)
        for src, city, score, txt in display_list:
            print("-", src, city or "", f"(final_score={score:.4f})")
            print("  ", txt[:300].replace("\n", " "))
        print("-" * 40)
