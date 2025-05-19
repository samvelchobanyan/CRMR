# train_lora.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

# ─── 1) Dataset ─────────────────────────────────────────────────────────────────
ds = load_dataset("json", data_files="training_data.jsonl", split="train")

# ─── 2) Tokenizer ────────────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False,
    trust_remote_code=True
)

# ─── Pad token fix ───────────────────────────────────────────────────────────────
# Llama-2 chat tokenizer has no pad_token by default; use eos_token instead
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ─── 3) Preprocessing ────────────────────────────────────────────────────────────
def preprocess(examples):
    inputs = tokenizer(
        examples["prompt"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    # switch to target tokenizer for labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["completion"],
            truncation=True,
            max_length=256,
            padding="max_length"
        )
    inputs["labels"] = labels["input_ids"]
    return inputs

train_ds = ds.map(
    preprocess,
    batched=True,
    remove_columns=["prompt", "completion"]
)

# ─── 4) Model (8-bit on GPU) ──────────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    use_auth_token=True
)

# ensure the pad_token is in the model embeddings
model.resize_token_embeddings(len(tokenizer))

# ─── 5) LoRA config ──────────────────────────────────────────────────────────────
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05
)
model = get_peft_model(model, peft_config)

# ─── 6) Training arguments ───────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="fine-tuned-loan-extractor",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_steps=500,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    tokenizer=tokenizer
)

# ─── 7) Train & save ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    trainer.train()
    model.save_pretrained("fine-tuned-loan-extractor")
    tokenizer.save_pretrained("fine-tuned-loan-extractor")
