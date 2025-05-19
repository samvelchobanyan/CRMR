# ai_service.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# 1) Create your FastAPI app
app = FastAPI(title="Local Loan‑Term Extractor")

# 2) Load a Hugging Face QA model locally (CPU or GPU)
qa = pipeline(
  task="question-answering",
  model="deepset/roberta-base-squad2",  # or another model you prefer
  device=-1  # set to 0 if you have a GPU
)

class ExtractRequest(BaseModel):
  field: str
  text: str

@app.post("/extract")
async def extract(req: ExtractRequest):
  # 3) Ask the model to find the answer for that field
  res = qa(question=req.field, context=req.text)
  return { "value": res.get("answer", "").strip() or "-" }
