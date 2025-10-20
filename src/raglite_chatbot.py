# src/raglite_chatbot.py
"""
raglite_chatbot.py
------------------
Retrieval-augmented chatbot with automatic LoRA detection and provenance-aware prompts.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from retrieval_faiss import TravelRetriever  # adjust import path if necessary

# Config
BASE_MODEL = "google/flan-t5-small"
LORA_DIR = "lora_travel_t5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and base model
print("🔹 Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
# Attach LoRA adapter if present
def load_model_with_optional_peft(base_model, adapter_dir):
    if os.path.isdir(adapter_dir) and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        print(f"✅ LoRA adapter detected in '{adapter_dir}' — loading adapter.")
        model_peft = PeftModel.from_pretrained(base_model, adapter_dir)
        return model_peft
    else:
        print("No LoRA adapter found: using base model only.")
        return base_model

model = load_model_with_optional_peft(base_model, LORA_DIR)
model.to(DEVICE)
model.eval()

# Initialize retriever
retriever = TravelRetriever("data/faq_data.json", "data/destinations.json", verbose=False)

# Prompting template (RAG)
PROMPT_TEMPLATE = (
    "You are a helpful travel assistant. Use ONLY the provided context to answer the question. "
    "If the information is not present in the context, say 'I don't know'. "
    "Be concise (2-3 sentences) and show the most relevant evidence line if applicable.\n\n"
    "CONTEXTS:\n{contexts}\n\nQUESTION: {question}\nANSWER:"
)

# Generation defaults (use same as baseline for fair comparison)
GEN_KWARGS = {
    "num_beams": 4,
    "max_new_tokens": 120,
    "do_sample": False,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
}

def format_contexts_for_prompt(items):
    """
    items: list of dicts with keys: text, meta, score, index, ce_score (optionally), combined_score, final_score
    Returns a string with numbered contexts including provenance.
    """
    lines = []
    for i, it in enumerate(items, start=1):
        meta = it.get("meta", {})
        src = meta.get("source", "unknown")
        src_id = meta.get("id", "")
        city = meta.get("city", "")
        # short text preview (avoid huge prompts)
        text = it.get("text", "").strip()
        # build provenance header
        prov = f"[{src}#{src_id}{(' | ' + city) if city else ''} | score={it.get('final_score', it.get('combined_score', it.get('score', None))):.3f}]"
        lines.append(f"{i}. {prov}\n{text}")
    return "\n\n".join(lines)

def rag_answer(query, top_k=3):
    # Get structured contexts
    contexts = retriever.retrieve_city_focused(query, k=top_k)
    if not contexts:
        contexts = retriever.retrieve(query, k=top_k)  # fallback to generic
    contexts_str = format_contexts_for_prompt(contexts)

    prompt = PROMPT_TEMPLATE.format(contexts=contexts_str, question=query)

    # Tokenize and move to device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(**inputs, **GEN_KWARGS)

    answer = tokenizer.decode(generated[0], skip_special_tokens=True)
    # Return answer and contexts (for logging / UI)
    return answer, contexts

# CLI
if __name__ == "__main__":
    print("🧭 Travel Assistant Chatbot (RAG-Lite + LoRA Ready)")
    print("Type 'exit' to quit.\n")
    while True:
        q = input("Ask a travel question: ")
        if q.strip().lower() == "exit":
            break
        ans, ctxs = rag_answer(q)
        print("\n[Answer]\n", ans)
        print("\n[Top contexts used]")
        for c in ctxs:
            print("-", c["meta"].get("source"), c["meta"].get("city", ""), f"(final_score={c.get('final_score'):.4f})")
            print("  ", c["text"][:300].replace("\n", " "))
        print("-" * 40)
