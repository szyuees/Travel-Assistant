# Travel Assistant Chatbot (DSA4213 Project)

## Overview
This project implements a **lightweight Retrieval-Augmented Generation (RAG)** chatbot designed to answer **travel-related FAQs** such as visa requirements, transport, attractions, and travel tips.  
It uses **Flan-T5 Small** as a base model for natural-language generation and combines it with a **FAISS semantic retriever** built from a curated knowledge base.  

To improve domain specificity without full fine-tuning, we apply **LoRA (Low-Rank Adaptation)** — a parameter-efficient fine-tuning (PEFT) method that updates only a small subset of model weights.  

The final system demonstrates:
-  *Prompt-only baseline model*  
-  *Retrieval-Augmented Generation (RAG-Lite) pipeline*  
-  *LoRA fine-tuned version for domain adaptation*  
-  *Automatic evaluation using semantic similarity*  
-  *Error analysis on low-performing outputs*

---

## Project Structure
| Folder | Description |
|--------|--------------|
| `data/` | JSON/CSV knowledge base, FAQs, and evaluation queries. |
| `src/` | All source code — model scripts, retrieval functions, and evaluation tools. |
| `reports/` | Project documentation and final report. |
| `docs/` | Diagrams and presentation slides. |

---

## Setup
### Clone the repository
```bash
git clone https://github.com/dhyyunn/DSA4213-group-project-travel-assistant-chatbot.git
cd DSA4213-group-project-travel-assistant-chatbot
```
### Create a virtual environment
``` bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Run the Chatbot
### a) Baseline (prompt-only)
Generates answers directly from Flan-T5 Small with no retrieval context.
```bash
python3 src/baseline_flanT5.py
```
### b) Retrieval-Augmented Chatbot (RAG-Lite)
Uses FAISS retrieval from both faq_data.json and destinations.json.
Automatically detects and loads LoRA weights (lora_travel_t5/) if available.
```bash
python3 src/raglite_chatbot.py
```
Example:
Ask a travel question: What should I visit in Paris?

[Retrieved Context]
Paris. Transport: Paris’s Metro and RER trains...
Attractions: Eiffel Tower, Louvre Museum, Notre-Dame Cathedral...
---------------
Answer: You can visit famous landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral in Paris.

### c) LoRA Fine-Tuning (Run in Google Colab)
This uses data/lora_train.csv (10–15 Q-A pairs).
Output will be saved in /lora_travel_t5/.
To re-use locally, the chatbot automatically loads this folder.
```bash
python3 src/lora_finetune.py
```

## Evaluation
### Automatic Evaluation
```bash
python3 src/evaluation.py
```
Compares:
	•	Baseline (Flan-T5 Small)
	•	RAG-Lite (Flan-T5 Small)
	•	RAG-Lite + LoRA (Fine-tuned)

Example Output:
Average similarity:
Sim_Baseline    0.63
Sim_RAG         0.84
Sim_LoRA        0.88
🎯 RAG-Lite with LoRA shows the highest factual accuracy.

### Error Analysis
We manually examined 10 lowest-similarity outputs.
Common failure types:
	•	Retrieval mismatch – retrieved wrong topic or city.
	•	Model underfitting – short or generic replies (“Paris”).
	•	Data sparsity – too small dataset with limited scope

## Contributors
• Deng Haoyun
• See Sze Yui
• Li Xiaoyue