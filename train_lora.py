# train_lora.py

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType

# ─── 1) Load dataset ─────────────────────────────────────────────────────────────
ds = load_dataset("json", data_files="training_data.jsonl", split="train")

# ─── 2) Tokenizer + pad fix ───────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ─── 3) Preprocessing ────────────────────────────────────────────────────────────
def preprocess(examples):
    out = {"input_ids": [], "labels": []}

    for prompt, completion in zip(examples["prompt"], examples["completion"]):
        # tokenize prompt and completion separately
        tokp = tokenizer(prompt, add_special_tokens=False)
        tokc = tokenizer(completion, add_special_tokens=False)

        # concatenate and truncate to 1024
        ids = tokp["input_ids"] + tokc["input_ids"]
        ids = ids[:1024]

        # build labels: mask prompt portion
        labs = [-100] * len(tokp["input_ids"]) + tokc["input_ids"]
        labs = labs[:1024]

        # pad both to 1024
        pad_len = 1024 - len(ids)
        ids += [tokenizer.pad_token_id] * pad_len
        labs += [-100] * pad_len

        out["input_ids"].append(ids)
        out["labels"].append(labs)

    # convert to proper batches/tensors
    batch = {
        "input_ids": out["input_ids"],
        "labels": out["labels"],
        "attention_mask": [
            [1 if token_id != tokenizer.pad_token_id else 0 for token_id in seq]
            for seq in out["input_ids"]
        ]
    }
    return batch

train_ds = ds.map(
    preprocess,
    batched=True,
    remove_columns=["prompt", "completion"]
)

# ─── 4) Load model in 8-bit on GPU ────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    use_auth_token=True
)
model.resize_token_embeddings(len(tokenizer))

# ─── 5) Configure LoRA ───────────────────────────────────────────────────────────
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05
)
model = get_peft_model(model, peft_config)

# ─── 6) TrainingArguments ────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="fine-tuned-loan-extractor",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_steps=500,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    label_pad_token_id=-100,
)

# ─── 7) Trainer & launch ─────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    tokenizer=tokenizer,
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained("fine-tuned-loan-extractor")
    tokenizer.save_pretrained("fine-tuned-loan-extractor")
