# train_lora_cpu.py

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType

# ─── 1) Locate training data ───────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE, "training_data.jsonl")
ds = load_dataset("json", data_files=data_path, split="train")

# ─── 2) Tokenizer ──────────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False,
    trust_remote_code=True
)

# ─── 3) Preprocessing ──────────────────────────────────────────────────────────
def preprocess(example):
    enc = tokenizer(
        example["prompt"],
        truncation=True,
        max_length=1024
    )
    with tokenizer.as_target_tokenizer():
        lbl = tokenizer(
            example["completion"],
            truncation=True,
            max_length=256
        )
    enc["labels"] = lbl["input_ids"]
    return enc

train_ds = ds.map(
    preprocess,
    batched=True,
    remove_columns=["prompt", "completion"]
)

# ─── 4) Load model ON CPU, full-precision ───────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True    # reduces peak RAM
)

# ─── 5) Attach LoRA adapters ────────────────────────────────────────────────────
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05
)
model = get_peft_model(model, peft_config)

# ─── 6) Training arguments for CPU ──────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="fine-tuned-loan-extractor",
    per_device_train_batch_size=1,     # keep tiny on CPU
    gradient_accumulation_steps=4,     # accumulate to simulate a larger batch
    gradient_checkpointing=True,       # trade compute for lower memory
    max_steps=500,
    learning_rate=3e-4,
    fp16=False,                        # no fp16 on CPU
    bf16=False,                        # no bf16 on CPU
    logging_steps=10,
    save_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    tokenizer=tokenizer
)

# ─── 7) Train ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("⚠️  Training on CPU only; this will be slow!")
    trainer.train()
    model.save_pretrained("fine-tuned-loan-extractor")
    tokenizer.save_pretrained("fine-tuned-loan-extractor")
