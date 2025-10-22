# ==========================================
# baseline_llama3.2.py
# Baseline model evaluation without retrieval
# ==========================================

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from sentence_transformers import SentenceTransformer, util
import pandas as pd, matplotlib.pyplot as plt

def evaluate_baseline(
    data_path="data/evaluation_queries.csv",
    save_path="reports/evaluation_llama3.2_baseline.csv"
):
    """
    Evaluates Llama-3.2-3B-Instruct on travel QA dataset without retrieval.
    Computes similarity between model answers and reference answers.
    """

    # ✅ Hugging Face login (optional, skip if already authenticated)
    try:
        login(new_session=False)
    except Exception:
        print("⚠️ Skipping Hugging Face login (already authenticated).")

    # ✅ Load model
    print("🔹 Loading model...")
    model_name = "meta-llama/Llama-3.2-3b-instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=bnb_config,
        use_auth_token=True,
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
    )

    # ✅ Load evaluation dataset
    df = pd.read_csv(data_path)
    print(f"🔹 Loaded {len(df)} evaluation queries.")

    def generate_answer(prompt):
        return generator(prompt)[0]["generated_text"].strip()

    df["generated_answer"] = df["query"].apply(generate_answer)

    # ✅ Compute semantic similarity
    print("🔹 Computing semantic similarity with BGE...")
    encoder = SentenceTransformer("BAAI/bge-large-en-v1.5")

    def compute_similarity(a, b):
        ea = encoder.encode(a, convert_to_tensor=True)
        eb = encoder.encode(b, convert_to_tensor=True)
        return float(util.cos_sim(ea, eb))

    df["similarity"] = df.apply(
        lambda row: compute_similarity(row["generated_answer"], row["expected_answer"]),
        axis=1,
    )

    avg_sim = df["similarity"].mean()
    print(f"✅ Average similarity: {avg_sim:.4f}")

    # ✅ Save results
    df.to_csv(save_path, index=False)
    print(f"💾 Results saved to {save_path}")

    # ✅ Visualize
    plt.figure(figsize=(10,4))
    plt.bar(range(len(df)), df["similarity"], color="cornflowerblue")
    plt.title(f"Baseline Similarity per Query (Avg={avg_sim:.3f})")
    plt.xlabel("Query Index")
    plt.ylabel("Cosine Similarity")
    plt.tight_layout()
    plt.show()

    return df


if __name__ == "__main__":
    evaluate_baseline()