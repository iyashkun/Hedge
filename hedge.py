import os, threading, time
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling, pipeline
import gradio as gr

MODEL_NAME = "gpt2"
MODEL_DIR = "./hf_model"
DATA_FILE = "train.txt"

os.makedirs(MODEL_DIR, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR if os.path.exists(f"{MODEL_DIR}/pytorch_model.bin") else MODEL_NAME)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def chat(prompt):
    with open(DATA_FILE, "a", encoding="utf-8") as f: f.write(prompt + "\n")
    return generator(prompt, max_new_tokens=100)[0]["generated_text"]

def train():
    if not os.path.exists(DATA_FILE): return
    dataset = TextDataset(tokenizer=tokenizer, file_path=DATA_FILE, block_size=128)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=MODEL_DIR, overwrite_output_dir=True, per_device_train_batch_size=4, num_train_epochs=1),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=dataset,
    )
    trainer.train()
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

def train_every_hour():
    while True:
        train()
        time.sleep(3600)

threading.Thread(target=train_every_hour, daemon=True).start()
gr.Interface(fn=chat, inputs="text", outputs="text", title="Hedge AI").launch()
