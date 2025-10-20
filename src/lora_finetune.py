# src/lora_finetune.py
"""
lora_finetune.py
Fine-tune Flan-T5-Small using LoRA for travel FAQ domain.

Improvements vs original:
- deterministic seed
- device awareness & fp16 if available
- robust LoRA target module detection with safety check
- label padding -> -100
- DataCollatorForSeq2Seq usage
- checkpointing, evaluation during training
- saves adapter and manifest for reproducibility
"""

import os
import json
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers import DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
import torch
import numpy as np

# -------------------
# Config (tune these)
# -------------------
MODEL_NAME = "google/flan-t5-small"
LORA_OUTPUT_DIR = "lora_travel_t5"
TRAIN_CSV = "data/lora_train.csv"
SEED = 42

# Training hyperparams (sensible defaults)
NUM_EPOCHS = 3
PER_DEVICE_BATCH = 2
GRAD_ACCUM_STEPS = 4     # effective batch = PER_DEVICE_BATCH * GRAD_ACCUM_STEPS
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 50
SAVE_STEPS = 200
EVAL_STEPS = 200
LOGGING_STEPS = 50
FP16 = torch.cuda.is_available()  # use mixed precision if GPU present

# LoRA hyperparams
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_BIAS = "none"

# -------------------
# Setup
# -------------------
os.environ["WANDB_DISABLED"] = "true"
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset("csv", data_files={"train": TRAIN_CSV})["train"]
dataset = dataset.train_test_split(test_size=0.1, seed=SEED)  # 90/10 split

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(device)

# -------------------
# Robust target_modules detection
# -------------------
# We attempt to find descriptive names for query/value projection layers.
# We'll search module names for likely substrings and print either suggestion or sample modules for manual inspection.
candidate_substrings = ["q", "v", "k", "q_proj", "v_proj", "key", "value", "query", "qkv", "wi", "wo"]

found_substrings = set()
sample_matches = []
for name, module in model.named_modules():
    lname = name.lower()
    for sub in candidate_substrings:
        if sub in lname:
            found_substrings.add(sub)
            sample_matches.append(name)
# If we found the minimal 'q' and 'v' substrings, prefer them.
if "q" in found_substrings and "v" in found_substrings:
    target_modules = ["q", "v"]
else:
    # fallback: pick up to 3 discovered substrings (still better than empty)
    target_modules = list(found_substrings)[:3]

if len(target_modules) == 0:
    # If zero, print sample module names and raise explicit error to instruct manual selection.
    print("=== Sample module names (first 200) ===")
    for i, (n, _) in enumerate(list(model.named_modules())[:200]):
        print(i, n)
    raise RuntimeError(
        "Unable to auto-detect LoRA target modules. Inspect sample module names above and set `target_modules` manually."
    )

print(f"Detected target_modules substrings for LoRA: {target_modules}")
# -------------------
# Attach LoRA
# -------------------
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=target_modules,
    lora_dropout=LORA_DROPOUT,
    bias=LORA_BIAS,
    task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # nice summary: ensures only adapter params are trainable

# -------------------
# Tokenize & preprocess
# -------------------
MAX_IN_LEN = 128
MAX_OUT_LEN = 64


def preprocess(batch):
    # expects columns: 'question' and 'answer'
    inputs = [f"Question: {q}" for q in batch["question"]]
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=MAX_IN_LEN)

    labels = tokenizer(batch["answer"], truncation=True, padding="max_length", max_length=MAX_OUT_LEN)
    labels_ids = labels["input_ids"]
    # replace pad token id's with -100 for loss ignoring
    labels_ids = [[(tid if tid != tokenizer.pad_token_id else -100) for tid in l] for l in labels_ids]
    model_inputs["labels"] = labels_ids
    return model_inputs


tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# Data collator (will also pad dynamically)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# -------------------
# TrainingArguments & Trainer
# -------------------
training_args = TrainingArguments(
    output_dir="lora_output",
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    logging_steps=LOGGING_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    fp16=FP16,
    remove_unused_columns=False,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=data_collator,
)

# -------------------
# Train + Save Adapter
# -------------------
print("=== Starting LoRA training ===")
trainer.train()

# Save adapter with PEFT-friendly save
print(f"Saving LoRA adapter to {LORA_OUTPUT_DIR} ...")
os.makedirs(LORA_OUTPUT_DIR, exist_ok=True)
model.save_pretrained(LORA_OUTPUT_DIR)

# Save small manifest for reproducibility
manifest = {
    "base_model": MODEL_NAME,
    "lora_config": {
        "r": LORA_R,
        "alpha": LORA_ALPHA,
        "target_modules": target_modules,
        "dropout": LORA_DROPOUT,
    },
    "training_args": {
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": PER_DEVICE_BATCH,
        "gradient_accum_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
    },
    "seed": SEED,
    "saved_at": datetime.utcnow().isoformat() + "Z",
}

with open(os.path.join(LORA_OUTPUT_DIR, "adapter_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("LoRA fine-tuning complete! Adapter saved to:", LORA_OUTPUT_DIR)
