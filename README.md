# Travel Assistant Chatbot (DSA4213 Project)

## Overview
This project implements a Retrieval-Augmented Generation (RAG)–based Travel Assistant Chatbot that can answer travel-related questions (e.g., transport, attractions, travel tips) using a lightweight open-source setup.

The chatbot integrates:
Llama-3.2-3B-Instruct as the generation backbone
BGE-Large-v1.5 as the semantic retriever
FAISS for efficient similarity search
A WikiVoyage knowledge base (≈200K chunks)

We further benchmark *Baseline*, *RAG (Top-k)*, and *Reranked RAG* variants and evaluate performance using semantic similarity metrics.

### Important Note !!
The Python scripts in src/ are designed for evaluation and experimentation (not as interactive chatbots).
For ease of reproduction, please use the notebook notebooks/testing_llama3.2.ipynb, which runs all experiments sequentially and generates the evaluation results under reports/.


---

## Project Structure
| Folder | Description |
|--------|--------------|
| `data/` | JSONL knowledge base, FAQs, and evaluation queries. But the dataset was too large, so we removed it here. |
| `src/` | All source code to evaluate models perfomance. |
| `reports/` | Project documentation and final report. |
| `notebooks/` | The notebook to run this project entirely. |


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

## How to run
### a) open the notebook
Open notebooks/testing_llama3.2.ipynb in Google Colab or VS Code Jupyter.

### b) Set Runtime
Change your runtime to GPU

### c) Run Sequentially
Run the notebook top to bottom — it includes:
1.	Environment setup & imports
2.	Load model + retriever
3.	Build FAISS index
4.	Run Baseline
5.	Run RAG (Top-k)
6.	Run RAG (Reranked)
7.	Generate evaluation reports
8.	Visualize results

All generated results are saved to reports/.


## Output Files
| File| Description |
|--------|--------------|
| `evaluation_llama3.2_baseline.csv` | Model only QA results.|
| `evaluation_llama3.2_RAG.csv` | RAG Top-k (k=3) results |
| `evaluation_llama3.2_RAG_reranked.csv` | RAG with reranking results |
| `reports/` | The notebook to run this project entirely. |

