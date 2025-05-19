import os
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

# 1. Set your model directory (where train_lora.py saved it)
MODEL_DIR = "fine-tuned-loan-extractor"

# 2. Load tokenizer & base model + LoRA adapter
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    load_in_8bit=True
)
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()

# 3. A helper that prompts the model to extract one field
def extract_field(text: str, field_name: str) -> str:
    prompt = f"Extract the **{field_name}** from the following loan terms document. " \
             f"Only output the value (or 'N/A' if missing):\n\n{text}\n\nValue:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    gen_cfg = GenerationConfig(
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )
    out = model.generate(**inputs, generation_config=gen_cfg)
    return tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

# 4. The list of fields to extract
FIELDS = [
    "Վարկի առավելագույն գումար",
    "Վարկի նվազագույն գումար",
    "Վարկի արժույթը",
    "Տարեկան անվանական տոկոսադրույք"
]

if __name__ == "__main__":
    # 5. Prepare CSV
    with open("extractions.csv", "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["filename"] + FIELDS)

        # 6. Loop over all .txt in your Outputs folder
        for root, _, files in os.walk("Outputs"):
            for fn in files:
                if not fn.endswith(".txt"):
                    continue
                path = os.path.join(root, fn)
                text = open(path, encoding="utf-8").read()
                row = [fn]
                for field in FIELDS:
                    try:
                        val = extract_field(text, field)
                    except Exception as e:
                        val = f"Error: {e}"
                    row.append(val)
                writer.writerow(row)

    print("Done! See extractions.csv")
