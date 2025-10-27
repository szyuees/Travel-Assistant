# ==========================================
# baseline_gemma2_2b_it.py
# Baseline model evaluation without retrieval
# ==========================================

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from sentence_transformers import SentenceTransformer, util
import pandas as pd, matplotlib.pyplot as plt

def evaluate_baseline(
    data_path="data/evaluation_queries.csv",
    save_path="reports/evaluation_gemma2_baseline.csv"   
):
    try:
        login(new_session=False)
    except Exception:
        print("⚠️ Skipping Hugging Face login (already authenticated).")

    print("🔹 Loading model...")
    model_name = "google/gemma-2-2b-it"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto",
        quantization_config=bnb_config, use_auth_token=True
    )
    generator = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=300, temperature=0.7, top_p=0.9,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
    )

    df = pd.read_csv(data_path)
    print(f"🔹 Loaded {len(df)} evaluation queries.")

    def generate_answer(user_prompt: str) -> str:
        messages = [
        {"role": "user", "content": f"You are a helpful travel/study assistant.\n\n{user_prompt}"}
        ]
        templ = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = generator(templ)[0]["generated_text"]
        return out[len(templ):].strip()

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
