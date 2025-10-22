#load data
import json, random

path = "data/wikivoyage_chunks.jsonl"

data = [json.loads(line) for line in open(path)]
print("Total entries:", len(data))

# sample 10 000 chunks for FAISS (balance quality vs. memory)
sampled_data = random.sample(data, 10_000)
corpus = [x["text"] for x in sampled_data]

# Build FAISS retrieval index
from sentence_transformers import SentenceTransformer
import numpy as np, faiss, torch

retriever = SentenceTransformer("BAAI/bge-large-en-v1.5")
embeddings = retriever.encode(corpus, convert_to_numpy=True, show_progress_bar=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings, dtype="float32"))
print("✅ Indexed:", index.ntotal, "WikiVoyage chunks.")