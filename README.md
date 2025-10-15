# Travel Assistant Chatbot (DSA4213 Project)

## Overview
This project builds a **lightweight, domain-specific chatbot** for answering travel FAQs and destination queries efficiently.  
It demonstrates **Retrieval-Augmented Generation (RAG)** and **prompt-based learning** using small pretrained models (Flan-T5 small).

### Key Features
- **Prompt-only baseline** using Flan-T5 Small.
- **RAG-lite chatbot** with FAISS retrieval on a curated travel knowledge base.
- **Evaluation** via semantic similarity (cosine) and human ratings.
- **Compact & efficient**, designed for low compute environments.

---

## Project Structure
| Folder | Description |
|--------|--------------|
| `data/` | JSON/CSV knowledge base, FAQs, and evaluation queries. |
| `src/` | All source code — model scripts, retrieval functions, and evaluation tools. |
| `notebooks/` | Data exploration and evaluation notebooks. |
| `reports/` | Project documentation and final report. |
| `docs/` | Diagrams and presentation slides. |

---

## Setup Instructions and how to run the model
### 1. Create a virtual environment
### 2. Download the required packages from requirements.txt
### 3. run the baseline model
```bash
python3 src/baseline_flanT5.py
```
### 4. run the raglite chatbot
```bash
python3 src/retrieval_faiss.py
python3 src/raglite_chatbot.py
```
