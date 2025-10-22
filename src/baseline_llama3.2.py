from huggingface_hub import login
login(new_session=False)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

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


def generate_answer(prompt):
    output = generator(prompt)[0]["generated_text"]
    return output.strip()

df["generated_answer"] = df["query"].apply(generate_answer)
df.head()


# test out performance of baseline model using BGE (a model to compute similarity score between expected answer and generated ones)
from sentence_transformers import SentenceTransformer, util

encoder = SentenceTransformer("BAAI/bge-large-en-v1.5")

def compute_similarity(a, b):
    ea = encoder.encode(a, convert_to_tensor=True)
    eb = encoder.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(ea, eb))

df["similarity"] = df.apply(
    lambda row: compute_similarity(row["generated_answer"], row["expected_answer"]),
    axis=1,
)

print("Average similarity:", df["similarity"].mean())



# visualise the result
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
plt.bar(range(len(df)), df["similarity"])
plt.title("Baseline Similarity per Query (Llama-3.2-3B-Instruct)")
plt.xlabel("Query Index")
plt.ylabel("Cosine Similarity")
plt.show()