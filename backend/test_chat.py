"""Test chat template generation"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
model.eval()

question = "What is the capital of France?"

# Try chat template
messages = [{"role": "user", "content": question}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"\nChat template prompt:\n{prompt}")

inputs = tokenizer(prompt, return_tensors="pt")
print(f"Input tokens: {inputs['input_ids'].shape[1]}")

with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print(f"\nFull output:\n{full_text}")
print(f"\nAnswer only:\n{answer}")
