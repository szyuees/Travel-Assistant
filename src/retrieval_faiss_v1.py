"""
retrieval_faiss.py
------------------
Unified retriever for both FAQs and destination cards.
Supports semantic retrieval, keyword re-ranking, and
city-based prioritization for travel-related questions.
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder


class TravelRetriever:
    def __init__(
        self,
        faq_path="data/faq_data.json",
        dest_path="data/destinations.json",
        model_name="all-MiniLM-L6-v2",
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"),
    ):
        print("🔹 Initializing TravelRetriever...")
        self.retriever = SentenceTransformer(model_name)

        # Load FAQ Data 
        with open(faq_path) as f:
            faq_data = json.load(f)
        faq_entries = [
            item["question"] + " " + item["answer"] for item in faq_data
        ]

        # Load Destination Data 
        with open(dest_path) as f:
            dest_data = json.load(f)
        dest_entries = []
        for d in dest_data:
            text_block = (
                f"{d['city']}. "
                f"Transport: {d['transport']}. "
                f"Attractions: {', '.join(d['attractions'])}. "
                f"Tips: {', '.join(d['tips'])}."
            )
            dest_entries.append(text_block)

        # Combine both datasets into one corpus
        self.corpus = []
        self.metadatas = []

        # add FAQ entries
        for i, item in enumerate(faq_data):
            text = item.get("question","") + " " + item.get("answer","")
            self.corpus.append(text)
            self.metadatas.append({"source":"faq", "source_file": faq_path, "id": i, "orig": item})

        # add destination entries
        for i, d in enumerate(dest_data):
            text_block = (
                f"{d.get('city','')}. "
                f"Transport: {d.get('transport','')}. "
                f"Attractions: {', '.join(d.get('attractions',[]))}. "
                f"Tips: {', '.join(d.get('tips',[]))}."
            )
            self.corpus.append(text_block)
            self.metadatas.append({"source":"dest", "source_file": dest_path, "id": i, "city": d.get('city',''), "orig": d})

        self.embeddings = self.retriever.encode(self.corpus, convert_to_numpy=True)

        # Build FAISS index 
        
        # Convert embeddings to numpy if needed
        import numpy as np
        emb = np.array(self.embeddings, dtype="float32")
        # L2-normalize rows to unit vectors for cosine similarity
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / (norms + 1e-12)
        self.embeddings = emb.astype("float32")

        # Use inner product on normalized vectors (equivalent to cosine similarity)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

        # List of supported cities (for detection)
        self.cities = [
            "Singapore", "Seoul", "Tokyo", "Bangkok", "Paris", "Los Angeles"
        ]

        print(f"✅ Loaded {len(self.corpus)} knowledge entries into FAISS index.")

    # --------------------------------------------------------------------
    def retrieve(self, query, k=3):
        """Default semantic + keyword retrieval."""
        q_emb = self.retriever.encode([query])
        D, I = self.index.search(np.array(q_emb, dtype="float32"), k * 3)
        # After index.search => D, I
        results = []
        for rank_idx, idx in enumerate(I[0]):
            score = float(D[0][rank_idx])
            results.append({
                "text": self.corpus[idx],
                "meta": self.metadatas[idx],
                "score": score,
                "index": idx
            })

        # Re-rank results (use improved re-ranker below), then return top-k results
        # after constructing 'results' list (semantic top K*3)
        pairs = [(query, r["text"]) for r in results]
        scores = self.cross_encoder.predict([p for p in pairs])  # returns scores
        for i, sc in enumerate(scores):
            results[i]["ce_score"] = float(sc)
        # final sort by ce_score (and maybe meta boosts)
        results = sorted(results, key=lambda r: r["ce_score"], reverse=True)
        return results[:k]

    # --------------------------------------------------------------------
    def retrieve_city_focused(self, query, k=3):
        """Detect city name and prioritize related entries."""
        q_emb = self.retriever.encode([query])
        D, I = self.index.search(np.array(q_emb, dtype="float32"), k * 5)
        candidates = [self.corpus[i] for i in I[0]]

        # Detect city name(s) in query
        detected = [c for c in self.cities if c.lower() in query.lower()]

        # Re-rank: city-matched first, then keyword overlap
        keywords = set(query.lower().split())
        ranked = sorted(
            candidates,
            key=lambda c: (
                any(city.lower() in c.lower() for city in detected),
                len(keywords & set(c.lower().split())),
            ),
            reverse=True,
        )
        return ranked[:k]


# ------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick test to verify functionality
    retriever = TravelRetriever()
    test_queries = [
        "What should I visit in Paris?",
        "How can I travel around Tokyo?",
        "When is the best time to visit Bangkok?"
    ]
    for q in test_queries:
        print(f"\n🔹 Query: {q}")
        results = retriever.retrieve_city_focused(q)
        for r in results:
            print(" -", r)