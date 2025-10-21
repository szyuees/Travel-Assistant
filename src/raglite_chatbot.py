#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG‑Lite Chatbot (sy-compatible) with optional JSONL KB loader
-----------------------------------------------------------------
• Works with existing FAISS retriever (retrieval_faiss.TravelRetriever) if present
• OR, when --kb_jsonl is provided, builds a lightweight in‑process BM25 retriever
• Safely loads LoRA adapter if available
• Token‑level context truncation to keep evidence
• Simple REPL for quick testing

Usage examples
--------------
# Use external JSONL KB (recommended for your new data pipeline)
python src/raglite_chatbot.py \
  --kb_jsonl data/processed/wikivoyage_chunks.jsonl \
  --model_name google/flan-t5-small \
  --top_k 3 --per_context_tokens 120

# Use existing FAISS retriever from the repo
python src/raglite_chatbot.py --use_faiss --top_k 3

Notes
-----
• JSONL each line should contain: {id,title,section,url,text, ...}
  (If you use a different content key like "content", it will be auto‑mapped.)
• LoRA directory can be passed via --lora_dir (defaults to lora_output/checkpoint-12)
• If CUDA is available, the model will move to GPU automatically.
"""

from __future__ import annotations
import os
import re
import sys
import json
import math
import time
import argparse
from typing import Iterable
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Optional PEFT/LoRA
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

# Optional BM25 for JSONL retriever fallback
try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except Exception:
    _HAS_BM25 = False

# Optional FAISS retriever from project
_HAS_FAISS = False
try:
    from retrieval_faiss import TravelRetriever  # project-local file
    _HAS_FAISS = True
except Exception:
    pass

# --------------------------
# Args
# --------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAG-Lite Chatbot (sy-compatible) with JSONL KB option")
    p.add_argument("--model_name", type=str, default=os.environ.get("MODEL_NAME", "google/flan-t5-small"),
                   help="HF model id for the generator")
    p.add_argument("--lora_dir", type=str, default=os.environ.get("LORA_DIR", "lora_output/checkpoint-12"),
                   help="Path to LoRA checkpoint directory (optional)")
    p.add_argument("--no_lora", action="store_true", help="Disable loading LoRA even if directory exists")

    # Retrieval control
    p.add_argument("--kb_jsonl", type=str, default=None,
                   help="Path to chunked KB in JSONL with fields like {id,title,section,url,text}")
    p.add_argument("--use_faiss", action="store_true", help="Force using project FAISS retriever (TravelRetriever)")
    p.add_argument("--top_k", type=int, default=3, help="#contexts to retrieve")

    # Context / generation
    p.add_argument("--per_context_tokens", type=int, default=120, help="Max tokens per context before concat")
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--no_sample", action="store_true", help="Disable sampling (beam search only)")

    # REPL
    p.add_argument("--no_repl", action="store_true", help="Only run once if --question provided")
    p.add_argument("--question", type=str, default=None, help="Optional single-turn question")

    return p

# --------------------------
# Data structures
# --------------------------

@dataclass
class Doc:
    text: str
    title: str = ""
    section: str = ""
    url: str = ""
    source: str = ""
    score: float = 0.0
    meta: Optional[Dict[str, Any]] = None

# --------------------------
# JSONL KB loader + BM25 retriever
# --------------------------

def load_kb_from_jsonl(path: str) -> List[Doc]:
    docs: List[Doc] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text") or obj.get("content") or obj.get("body") or ""
            if not text:
                continue
            docs.append(Doc(
                text=text,
                title=obj.get("title", ""),
                section=obj.get("section", ""),
                url=obj.get("url", obj.get("link", "")),
                source=obj.get("source", "JSONL"),
                score=0.0,
                meta=obj,
            ))
    return docs

class BM25Retriever:
    def __init__(self, docs: List[Doc]):
        if not _HAS_BM25:
            raise RuntimeError("rank_bm25 not installed. Please `pip install rank-bm25`.\n")
        import re
        self.docs = docs
        # very light tokenization
        tokenized = [self._simple_tok(d.text) for d in docs]
        self.bm25 = BM25Okapi(tokenized)

    _split_re = None
    def _simple_tok(self, s: str) -> List[str]:
        import re
        if self._split_re is None:
            self.__class__._split_re = re.compile(r"[\W_]+", re.UNICODE)
        return [t for t in self._split_re.split(s.lower()) if t]

    def retrieve(self, query: str, top_k: int = 3) -> List[Doc]:
        q_tok = self._simple_tok(query)
        scores = self.bm25.get_scores(q_tok)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        out: List[Doc] = []
        for i in idxs:
            d = self.docs[i]
            out.append(Doc(
                text=d.text,
                title=d.title,
                section=d.section,
                url=d.url,
                source=d.source or "JSONL",
                score=float(scores[i]),
                meta=d.meta,
            ))
        return out

# --------------------------
# Generation utilities
# --------------------------

def load_model(model_name: str, lora_dir: Optional[str], no_lora: bool) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    tok = AutoTokenizer.from_pretrained(model_name)
    if not hasattr(tok, "model_max_length") or tok.model_max_length is None:
        tok.model_max_length = 512
    else:
        tok.model_max_length = min(tok.model_max_length, 512)
    tok.truncation_side = "right"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if (not no_lora) and _HAS_PEFT and lora_dir and os.path.isdir(lora_dir):
        try:
            model = PeftModel.from_pretrained(model, lora_dir)
            print(f"[LoRA] Loaded adapter from: {lora_dir}")
        except Exception as e:
            print(f"[LoRA] Warning: failed to load adapter from {lora_dir}: {e}")
    else:
        if not _HAS_PEFT and lora_dir and os.path.isdir(lora_dir):
            print("[LoRA] peft not installed; skipping adapter load.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tok, model

PROMPT_TEMPLATE = (
    "Answer ONLY with bullet points using the EVIDENCE below. "
    "If not found in the evidence, answer exactly: I don't know based on the provided sources.\n\n"
    "Question: {question}\n\n"
    "EVIDENCE (title — URL, then a sentence):\n{context}\n\n"
    "Bulleted answer (2–4 bullets, each ends with (Title — URL)):\n"
)

def truncate_tokens(tok: AutoTokenizer, text: str, max_tokens: int) -> str:
    # Reserve 16 tokens margin for safety
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tok.decode(ids, skip_special_tokens=True)

# 句子切分/分词小工具
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_SPLIT = re.compile(r"[\W_]+")

def _tok(s: str) -> list[str]:
    return [t for t in _WORD_SPLIT.split(s.lower()) if t]

QUESTION_HINTS = {
    "see": {"see","sight","landmark","museum","cathedral","gallery","park","old","historic"},
    "do": {"do","activity","tour","boat","canal","binnendieze","walk","hike","market","festival","play","fun"},
    "transport": {"transport","metro","subway","bus","train","tram","get","around","pass","ticket"}
}

def _hint_set(question: str) -> set[str]:
    q = question.lower()
    if any(k in q for k in ["transport","get around","metro","subway","bus","train","tram"]):
        return QUESTION_HINTS["transport"]
    if any(k in q for k in ["what to do","things to do","play","fun","activity"]):
        return QUESTION_HINTS["do"]
    if any(k in q for k in ["visit","see","must see","attraction"]):
        return QUESTION_HINTS["see"]
    return set()

def select_evidence_sentences(question: str, docs: list[Doc], max_total_sents: int = 6) -> list[tuple[str, Doc]]:
    q_tokens = set(_tok(question))
    hints = _hint_set(question)
    cands: list[tuple[float, str, Doc]] = []
    for d in docs:
        for sent in _SENT_SPLIT.split(d.text):
            s = sent.strip()
            if not (40 <= len(s) <= 300):
                continue
            stoks = set(_tok(s))
            inter = len(q_tokens & stoks)
            if inter == 0:
                continue
            bonus = 0
            if hints and (hints & stoks):   # 意图词命中加分
                bonus += 2
            if any(w in stoks for w in ["city","centre","center","cathedral","canal","binnendieze","museum","pass","ticket","metro","train"]):
                bonus += 1
            cands.append((inter + bonus, s, d))
    cands.sort(key=lambda x: x[0], reverse=True)
    out, seen = [], set()
    for _, s, d in cands:
        if len(out) >= max_total_sents: break
        if s in seen: continue
        seen.add(s); out.append((s, d))
    return out


# -------- re-ranking & filtering helpers --------
CITY_SECTIONS = ("understand","get in","get around","see","do","buy","eat","drink","sleep")
BAD_TITLES = ("phrasebook","phrase book","phrase-book","trail","heritage","worship","airport","province","county","district")

def _tok_l(s: str) -> list[str]:
    return [t for t in re.split(r"[\W_]+", (s or "").lower()) if t]

def _guess_place_tokens(question: str) -> set[str]:
    # 从问句里抓诸如 "in/at/to/of Paris" 的地点词；不命中则退回整句粗分词
    q = question.strip()
    m = re.search(r"\b(in|at|to|of)\s+([A-Z][\w'’\- ]+)", q)
    cand = (m.group(2) if m else q)
    return set(_tok_l(cand))

def filter_and_rerank(docs: list[Doc], question: str) -> list[Doc]:
    q_terms = _tok_l(question)
    place = _guess_place_tokens(question)

    # 1) 先过滤明显无关（phrasebook 等），除非问句明确包含这些词
    filtered = []
    for d in docs:
        title_l = (d.title or "").lower()
        if any(b in title_l for b in BAD_TITLES) and not any(b in q_terms for b in BAD_TITLES):
            continue
        filtered.append(d)
    if not filtered:
        filtered = docs[:]

    # 2) 打分：原BM25分 + 标题命中 + 地名重合 + 城市章节加权
    def score(d: Doc) -> float:
        s = d.score
        title_l = (d.title or "").lower()
        title_toks = set(_tok_l(d.title))
        sect_l = (d.section or "").lower()
        s += 1.0 * sum(t in title_l for t in q_terms)          # 问句词击中标题
        if place and (place & title_toks):                      # 标题含地名
            s += 3.5
        if place and place.issubset(title_toks):                # 标题几乎等于地名
            s += 2.0
        if any(k in sect_l for k in CITY_SECTIONS):             # 常见城市章节优先
            s += 1.2
        return s

    filtered.sort(key=score, reverse=True)

    # 3) 去重（title+url）
    seen = set(); out = []
    for d in filtered:
        key = (d.title, d.url)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def build_context_block(docs: list[Doc], tok: AutoTokenizer, per_context_tokens: int, question: str = "") -> str:
    # 若提供了question，则优先走“句子级”证据压缩；否则退回原 chunk 级拼接
    if question:
        pairs = select_evidence_sentences(question, docs, max_total_sents=6)
        blocks = []
        for s, d in pairs:
            # 给每条证据附上来源行
            header = f"[{d.title or d.source}] {d.url or ''}"
            body = truncate_tokens(tok, s, per_context_tokens)
            blocks.append(f"{header}\n{body}")
        if blocks:
            return "\n\n".join(blocks)
    # 回退：按 chunk 拼接
    blocks = []
    for d in docs:
        header = f"[{d.title or d.source}] {d.url or ''} (score={d.score:.4f})\n"
        body = truncate_tokens(tok, d.text, per_context_tokens)
        blocks.append(header + body)
    return "\n\n".join(blocks)


@torch.inference_mode()
def generate_answer(model, tok, prompt: str, max_new_tokens: int, num_beams: int, do_sample: bool) -> str:
    device = model.device
    inputs = tok(
        prompt,
        return_tensors="pt",
        truncation=True,           # 关键：超长时截断
        max_length=min(getattr(tok, "model_max_length", 512), 512)
    ).to(device)

    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
 #      min_new_tokens=40,  
        num_beams=num_beams,
        do_sample=do_sample,
 #      length_penalty=1.05,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return tok.decode(gen[0], skip_special_tokens=True)

# --------------------------
# Main RAG flow
# --------------------------

def build_retriever(args) -> Tuple[Any, str]:
    """Return (retriever, mode_str). retriever must have .retrieve(query, top_k)->List[Doc] or List[dict]."""
    if args.kb_jsonl:
        docs = load_kb_from_jsonl(args.kb_jsonl)
        if not docs:
            raise SystemExit(f"[KB] No docs loaded from {args.kb_jsonl}")
        if not _HAS_BM25:
            raise SystemExit("[KB] rank_bm25 not installed. `pip install rank-bm25`.")
        retr = BM25Retriever(docs)
        return retr, "bm25-jsonl"

    # Fall back to FAISS retriever from project when requested or available
    if args.use_faiss:
        if not _HAS_FAISS:
            raise SystemExit("[FAISS] retrieval_faiss.TravelRetriever not available in this env.")
        return TravelRetriever(), "faiss"

    # Auto: prefer FAISS if available; else error out
    if _HAS_FAISS:
        return TravelRetriever(), "faiss"
    raise SystemExit("No retriever available. Provide --kb_jsonl or install project FAISS retriever.")


def normalize_docs(rets: List[Any]) -> List[Doc]:
    out: List[Doc] = []
    for r in rets:
        if isinstance(r, Doc):
            out.append(r)
        elif isinstance(r, dict):
            out.append(Doc(
                text=r.get("text") or r.get("content") or r.get("chunk") or "",
                title=r.get("title", ""),
                section=r.get("section", ""),
                url=r.get("url", r.get("link", "")),
                source=r.get("source", "retriever"),
                score=float(r.get("score", 0.0)),
                meta=r,
            ))
    filtered = []
    for d in out:
        t = (d.title or "").lower()
        if any(bad in t for bad in ["phrasebook", "phrase book", "phrase-book"]):
            continue
        filtered.append(d)
    # 若全被过滤，退回原 out
    keep = filtered if filtered else out
    # 重新按分数排序（降序）
    keep.sort(key=lambda x: x.score, reverse=True)
    return keep


def rag_answer(question: str, retriever, tok, model, top_k: int, per_context_tokens: int, gen_cfg: dict) -> Tuple[str, List[Doc]]:
    raw = retriever.retrieve(question, top_k=top_k)
    docs = normalize_docs(raw)
    docs = filter_and_rerank(docs, question)
    context = build_context_block(docs, tok, per_context_tokens, question=question)
    prompt = PROMPT_TEMPLATE.format(question=question.strip(), context=context)
    ans = generate_answer(model, tok, prompt, **gen_cfg)
    return ans, docs

# --------------------------
# REPL
# --------------------------

def main():
    args = build_parser().parse_args()

    print("[Init] Loading model ...")
    tok, model = load_model(args.model_name, args.lora_dir, args.no_lora)

    print("[Init] Building retriever ...")
    retriever, mode = build_retriever(args)
    print(f"[Init] Retriever mode: {mode}")

    gen_cfg = dict(max_new_tokens=args.max_new_tokens,
                   num_beams=args.num_beams,
                   do_sample=(not args.no_sample))

    if args.question:
        ans, ctxs = rag_answer(args.question, retriever, tok, model, args.top_k, args.per_context_tokens, gen_cfg)
        print("\n[Answer]\n" + ans)
        if ctxs:
            print("\n[Contexts]")
            for d in ctxs:
                print(f"- {d.title or d.source} | {d.url} | score={d.score:.4f}")
        return

    if args.no_repl:
        return

    print("Type 'exit' to quit.\n")
    while True:
        try:
            q = input("Ask a travel question: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.strip().lower() in {"exit", "quit"}:
            break
        if not q.strip():
            continue
        ans, ctxs = rag_answer(q, retriever, tok, model, args.top_k, args.per_context_tokens, gen_cfg)
        print("\n[Answer]\n" + ans)
        if ctxs:
            print("\n[Contexts]")
            for d in ctxs:
                print(f"- {d.title or d.source} | {d.url} | score={d.score:.4f}")
        print("-"*60)


if __name__ == "__main__":
    main()
