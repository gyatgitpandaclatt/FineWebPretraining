# Block 9 of 10 — samples text from the trained model on a fixed prompt.
# Depends on: block 4 (EFFECTIVE_MODEL), block 7 (trained_model).
import torch
from transformers import AutoTokenizer

_device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(EFFECTIVE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

prompt = "The process of photosynthesis begins when"
ids = tokenizer(prompt, return_tensors="pt").to(_device)
out = trained_model.generate(**ids, max_new_tokens=80, do_sample=True, temperature=0.8, top_p=0.95)
print(tokenizer.decode(out[0], skip_special_tokens=True))
