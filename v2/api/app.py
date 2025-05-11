from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "./hf_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

app = FastAPI()

class Prompt(BaseModel):
    text: str

@app.post("/chat")
def chat(prompt: Prompt):
    output = gen(prompt.text, max_new_tokens=100)[0]["generated_text"]
    return {"response": output}
