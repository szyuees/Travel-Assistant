"""
lora_finetune.py
Fine-tune Flan-T5-Small using LoRA for travel FAQ domain.
"""

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import pandas as pd

# Load small QA dataset
df = pd.read_csv("data/lora_train.csv")
dataset = load_dataset("csv", data_files={"train": "data/lora_train.csv"})
dataset = dataset["train"].train_test_split(test_size=0.2)

# Model & tokenizer
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

candidate_substrings = ["q", "k", "v", "q_proj", "v_proj", "qkv", "wi", "wo"]  # common candidates

matched = set()
for name, _ in model.named_modules():
    lowered = name.lower()
    for sub in candidate_substrings:
        if sub in lowered:
            matched.add(sub)

if "q" in matched and "v" in matched:
    target_modules = ["q", "v"]
else:
    # fallback: use the discovered substrings (keeps it conservative)
    target_modules = list(matched)[:3]  # pick up to 3 substrings

if len(target_modules) == 0:
    raise ValueError(
        "Unable to automatically determine target_modules for LoRA. "
        "Please inspect model.named_modules() and set target_modules manually."
    )

# LoRA configuration
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=target_modules,  # query/value projections
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, config)

# Tokenize
def preprocess(batch):
    inputs = [f"Question: {q}" for q in batch["question"]]
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=128)
    labels = tokenizer(batch["answer"], truncation=True, padding="max_length", max_length=64)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess, batched=True)

# Training
import os
os.environ["WANDB_DISABLED"] = "true"
args = TrainingArguments(
    output_dir="lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="no"
)
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("lora_travel_t5")
print("LoRA fine-tuning complete!")