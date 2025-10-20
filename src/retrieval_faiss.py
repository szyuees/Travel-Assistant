"""
retrieval_faiss.py — improved FAISS retriever with metadata, normalized embeddings,
cross-encoder re-ranking, normalized score combination, and correct city-focused boosting.

Key fixes vs previous version:
- Ensure scores of different origins (FAISS semantic & Cross-Encoder logits) are normalized
  (min-max within the candidate set) before combining.
- For city-focused retrieval, retrieve a larger candidate set (fetch_k) then apply boosting
  before final truncation to top-k.
- Save/load metadata flag 'embeddings_normalized' with corpus metadata to avoid ambiguity.
- Debug prints show raw and normalized scores to make behaviors transparent.
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

# Optional CrossEncoder import
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except Exception:
    CROSS_ENCODER_AVAILABLE = False

# Small stopword set used only for fallback overlap scoring
_SIMPLE_STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "is", "are", "it",
    "what", "where", "how", "do", "i", "you", "we", "they", "my", "your"
}


def _min_max_normalize(values: List[float]) -> List[float]:
    """
    Min-max normalize a list of floats to [0,1]; if all values the same, return 0.5 for each.
    Returns list of floats in [0,1].
    """
    if len(values) == 0:
        return []
    arr = np.array(values, dtype=float)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if abs(vmax - vmin) < 1e-12:
        # constant vector -> return midpoints to avoid all zeros
        return [0.5] * len(values)
    normed = (arr - vmin) / (vmax - vmin)
    return normed.tolist()


class TravelRetriever:
    def __init__(
        self,
        faq_path: str = "data/faq_data.json",
        dest_path: str = "data/destinations.json",
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str = "data/faiss_index.bin",
        meta_path: str = "data/corpus_metadata.json",
        embeddings_path: str = "data/embeddings.npy",
        use_cross_encoder: bool = True,
        verbose: bool = True,
    ):
        self.faq_path = faq_path
        self.dest_path = dest_path
        self.index_path = index_path
        self.meta_path = meta_path
        self.embeddings_path = embeddings_path
        self.verbose = verbose

        if self.verbose:
            print("🔹 Initializing TravelRetriever...")
            print(f" - Embedding model: {model_name}")

        self.retriever = SentenceTransformer(model_name)

        self.cross_encoder = None
        if use_cross_encoder and CROSS_ENCODER_AVAILABLE:
            try:
                if self.verbose:
                    print(" - Loading cross-encoder for re-ranking (may download model)...")
                # ms-marco mini is a good small cross-encoder for relevance scoring
                self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception as e:
                if self.verbose:
                    print("   Cross-encoder load failed, falling back to overlap-based re-ranker:", e)
                self.cross_encoder = None
        else:
            if use_cross_encoder and not CROSS_ENCODER_AVAILABLE and self.verbose:
                print(" - Cross-encoder not installed; using fallback re-ranker.")

        self.corpus: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        self.cities = set(["Singapore", "Seoul", "Tokyo", "Bangkok", "Paris", "Los Angeles"])

        # load if exists; else build fresh (and save)
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path) and os.path.exists(self.embeddings_path):
            if self.verbose:
                print(" - Found saved index and metadata, loading...")
            self._load_index_and_meta()
        else:
            if self.verbose:
                print(" - No saved index found — building index now...")
            self._build_corpus_and_index()

    # -------------------------
    # Build / Save / Load
    # -------------------------
    def _build_corpus_and_index(self):
        faq_data, dest_data = [], []
        if os.path.exists(self.faq_path):
            with open(self.faq_path, "r", encoding="utf-8") as f:
                faq_data = json.load(f)
        else:
            if self.verbose:
                print(f"⚠️ FAQ file not found at {self.faq_path}")

        if os.path.exists(self.dest_path):
            with open(self.dest_path, "r", encoding="utf-8") as f:
                dest_data = json.load(f)
        else:
            if self.verbose:
                print(f"⚠️ Destinations file not found at {self.dest_path}")

        self.corpus = []
        self.metadatas = []

        for i, item in enumerate(faq_data):
            q = item.get("question", "")
            a = item.get("answer", "")
            text = (q + " " + a).strip()
            self.corpus.append(text)
            self.metadatas.append({"source": "faq", "source_file": self.faq_path, "id": i, "text": text, "orig": item})

        for i, d in enumerate(dest_data):
            city = d.get("city", "")
            transport = d.get("transport", "")
            attractions = ", ".join(d.get("attractions", []))
            tips = ", ".join(d.get("tips", []))
            text_block = f"{city}. Transport: {transport}. Attractions: {attractions}. Tips: {tips}".strip()
            self.corpus.append(text_block)
            self.metadatas.append({"source": "destination", "source_file": self.dest_path, "id": i, "city": city, "text": text_block, "orig": d})
            if city:
                self.cities.add(city)

        if len(self.corpus) == 0:
            raise RuntimeError("Corpus empty. Check your data files.")

        if self.verbose:
            print(f" - Encoding {len(self.corpus)} entries with {self.retriever.__class__.__name__}...")
        emb = self.retriever.encode(self.corpus, convert_to_numpy=True, show_progress_bar=self.verbose)

        # L2-normalize rows to unit length for cosine similarity (we will use IndexFlatIP)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / (norms + 1e-12)
        self.embeddings = emb.astype("float32")

        dim = self.embeddings.shape[1]
        if self.verbose:
            print(f" - Building FAISS IndexFlatIP (dim={dim}) and adding embeddings...")
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        # Save both index and a metadata file that includes whether embeddings were normalized
        self._save_index_and_meta()

    def _save_index_and_meta(self):
        if self.verbose:
            print(" - Saving index, metadata, and embeddings to disk...")
        faiss.write_index(self.index, self.index_path)
        meta_dump = {
            "embeddings_normalized": True,
            "entries": self.metadatas
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dump, f, ensure_ascii=False, indent=2)
        np.save(self.embeddings_path, self.embeddings)

    def _load_index_and_meta(self):
        self.index = faiss.read_index(self.index_path)
        meta_dump = json.load(open(self.meta_path, "r", encoding="utf-8"))
        self.metadatas = meta_dump.get("entries", [])
        self.corpus = [m.get("text", "") for m in self.metadatas]
        try:
            self.embeddings = np.load(self.embeddings_path)
        except Exception:
            self.embeddings = None
        # update cities set
        for m in self.metadatas:
            if m.get("city"):
                self.cities.add(m["city"])

    # -------------------------
    # Low-level helpers
    # -------------------------
    def _query_encode(self, query: str) -> np.ndarray:
        q_emb = self.retriever.encode([query], convert_to_numpy=True)
        q_emb = q_emb.astype("float32")
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
        return q_emb

    def _fallback_overlap_score(self, query: str, doc_text: str) -> int:
        q_tokens = [t for t in query.lower().split() if t not in _SIMPLE_STOPWORDS]
        d_tokens = [t for t in doc_text.lower().split() if t not in _SIMPLE_STOPWORDS]
        return len(set(q_tokens) & set(d_tokens))

    def _cross_encoder_score(self, query: str, docs: List[str]) -> Optional[List[float]]:
        if self.cross_encoder is None:
            return None
        pairs = [(query, d) for d in docs]
        scores = self.cross_encoder.predict(pairs)
        return [float(s) for s in scores]

    # -------------------------
    # Main retrieval (generic)
    # -------------------------
    def retrieve(self, query: str, k: int = 3, fetch_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents for 'query'.
        - fetch_k controls how many neighbors to retrieve from FAISS (default: max(50, k*10)).
        - Returns structured list with these keys: text, meta, score (semantic), index,
          ce_score (optional), combined_score (normalized combination used for ranking).
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call _build_corpus_and_index() or ensure files exist.")

        if fetch_k is None:
            fetch_k = max(50, k * 10)

        q_emb = self._query_encode(query)
        D, I = self.index.search(q_emb, fetch_k)
        raw_scores = D[0]
        indices = I[0]

        candidates = []
        for pos, idx in enumerate(indices):
            if int(idx) < 0 or int(idx) >= len(self.corpus):
                continue
            candidates.append({
                "text": self.corpus[int(idx)],
                "meta": self.metadatas[int(idx)],
                "score": float(raw_scores[pos]),  # semantic similarity (inner product on normalized vectors)
                "index": int(idx)
            })

        if len(candidates) == 0:
            return []

        # Cross-encoder re-ranking (if available)
        docs = [c["text"] for c in candidates]
        ce_scores = self._cross_encoder_score(query, docs)

        # Attach ce_score or fallback ov_score
        if ce_scores is not None:
            for i, s in enumerate(ce_scores):
                candidates[i]["ce_score"] = float(s)
        else:
            for i, c in enumerate(candidates):
                candidates[i]["ov_score"] = float(self._fallback_overlap_score(query, c["text"]))

        # Normalize both semantic and CE/ov scores to [0,1] so they are comparable
        sem_vals = [c["score"] for c in candidates]
        sem_norm = _min_max_normalize(sem_vals)

        if ce_scores is not None:
            ce_vals = [c.get("ce_score", 0.0) for c in candidates]
            ce_norm = _min_max_normalize(ce_vals)
        else:
            ov_vals = [c.get("ov_score", 0.0) for c in candidates]
            ce_norm = _min_max_normalize(ov_vals)  # we keep name 'ce_norm' but it's overlap norm here

        # Weighted combination: give cross-encoder more weight if available (configurable)
        alpha = 0.7  # weight for ce_norm
        beta = 0.3   # weight for semantic norm

        for i, c in enumerate(candidates):
            c["sem_norm"] = float(sem_norm[i])
            c["ce_norm"] = float(ce_norm[i])
            c["combined_score"] = float(alpha * c["ce_norm"] + beta * c["sem_norm"])

        # Sort by combined_score descending and return top-k
        candidates = sorted(candidates, key=lambda r: r["combined_score"], reverse=True)

        # Ensure a consistent 'final_score' key exists (use combined_score as default)
        for c in candidates:
            if "final_score" not in c:
                c["final_score"] = float(c.get("combined_score", c.get("score", 0.0)))

        # Keep a consistent return structure (full candidates but trimmed to k)
        return candidates[:k]


    # -------------------------
    # City-focused retrieval
    # -------------------------
    def retrieve_city_focused(self, query: str, k: int = 3, fetch_k: Optional[int] = None,
                              city_boost_meta: float = 0.25, city_boost_text: float = 0.15) -> List[Dict[str, Any]]:
        """
        City-focused retrieval:
        - Detect cities via case-insensitive substring match.
        - Retrieve a larger pool (fetch_k default is large) via self.retrieve but request that
          method to return more candidates (we call internal variant below for full candidate pool).
        - Apply small additive boosts (scaled relative to normalized combined_score).
        - Sort by final_score (combined_score + boost) and return top-k.
        """
        # detect cities in query
        q_lower = query.lower()
        detected = [c for c in self.cities if c.lower() in q_lower]
        # If no detected cities, behave like generic retrieve
        if len(detected) == 0:
            return self.retrieve(query, k=k, fetch_k=fetch_k)

        # For city-focused: fetch a larger candidate pool (we want to give city docs a chance)
        if fetch_k is None:
            fetch_k = max(200, k * 50)  # large pool to surface city-matching docs
        # We need the larger, untrimmed candidate list with combined scoring. So slightly adapt retrieve:
        q_emb = self._query_encode(query)
        D, I = self.index.search(q_emb, fetch_k)
        raw_scores = D[0]
        indices = I[0]

        candidates = []
        for pos, idx in enumerate(indices):
            if int(idx) < 0 or int(idx) >= len(self.corpus):
                continue
            candidates.append({
                "text": self.corpus[int(idx)],
                "meta": self.metadatas[int(idx)],
                "score": float(raw_scores[pos]),
                "index": int(idx)
            })

        if len(candidates) == 0:
            return []

        # Re-rank via cross-encoder or fallback overlap
        docs = [c["text"] for c in candidates]
        ce_scores = self._cross_encoder_score(query, docs)
        if ce_scores is not None:
            for i, s in enumerate(ce_scores):
                candidates[i]["ce_score"] = float(s)
        else:
            for i, c in enumerate(candidates):
                candidates[i]["ov_score"] = float(self._fallback_overlap_score(query, c["text"]))

        # Normalize sem and ce/ov
        sem_vals = [c["score"] for c in candidates]
        sem_norm = _min_max_normalize(sem_vals)

        if ce_scores is not None:
            ce_vals = [c.get("ce_score", 0.0) for c in candidates]
            ce_norm = _min_max_normalize(ce_vals)
        else:
            ov_vals = [c.get("ov_score", 0.0) for c in candidates]
            ce_norm = _min_max_normalize(ov_vals)

        alpha = 0.7
        beta = 0.3

        for i, c in enumerate(candidates):
            c["sem_norm"] = float(sem_norm[i])
            c["ce_norm"] = float(ce_norm[i])
            c["combined_score"] = float(alpha * c["ce_norm"] + beta * c["sem_norm"])

        # Apply city boost: scaled additive boost (small values on [0,1] normalized scale)
        detected_set = set([d.lower() for d in detected])
        for c in candidates:
            meta = c.get("meta", {})
            city_in_meta = bool(meta.get("city") and meta.get("city", "").lower() in detected_set)
            city_in_text = any((dc in c.get("text", "").lower()) for dc in detected_set)
            boost = 0.0
            if city_in_meta:
                boost += city_boost_meta
            if city_in_text:
                boost += city_boost_text
            c["boost"] = float(boost)
            c["final_score"] = float(c["combined_score"] + boost)

        # sort by final_score and return top-k
        candidates = sorted(candidates, key=lambda r: r["final_score"], reverse=True)
        return candidates[:k]

    # -------------------------
    # Debug printing helper
    # -------------------------
    def debug_print_top_k(self, query: str, k: int = 3):
        print(f"\nQuery: {query}\n--- Generic retrieve (top {k}) ---")
        top = self.retrieve(query, k=k)
        for i, r in enumerate(top):
            print(f"Rank {i+1} | idx={r['index']} | src={r['meta'].get('source')} | sem={r['score']:.6f} | ce_norm={r.get('ce_norm'):.4f} | sem_norm={r.get('sem_norm'):.4f} | combined={r.get('combined_score'):.4f}")
            print(" Text:", r["text"][:200].replace("\n", " "), "\n")

        print(f"\n--- City-focused retrieve (top {k}) ---")
        topc = self.retrieve_city_focused(query, k=k)
        for i, r in enumerate(topc):
            print(f"Rank {i+1} | idx={r['index']} | src={r['meta'].get('source')} | sem={r['score']:.6f} | ce_norm={r.get('ce_norm'):.4f} | combined={r.get('combined_score'):.4f} | boost={r.get('boost'):.4f} | final={r.get('final_score'):.4f}")
            print(" Text:", r["text"][:200].replace("\n", " "), "\n")


# Example usage (run as script)
if __name__ == "__main__":
    retriever = TravelRetriever(verbose=True)
    examples = [
        "What should I visit in Paris?",
        "How can I travel around Tokyo?",
        "When is the best time to visit Bangkok?"
    ]
    for q in examples:
        retriever.debug_print_top_k(q, k=3)