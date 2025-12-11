"""Quick test to verify model generation"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
model.eval()

print("\nTesting generation...")
question = "What is the capital of France?"
inputs = tokenizer(question, return_tensors="pt")

print(f"Input: {question}")
print(f"Input tokens: {inputs['input_ids'].shape[1]}")

with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=30,
        min_new_tokens=5,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print(f"\nFull output: {full_text}")
print(f"Answer only: {answer}")
print(f"Generated {outputs.shape[1] - inputs['input_ids'].shape[1]} new tokens")
