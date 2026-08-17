# Block 10 of 10 — saves the trained model + tokenizer to OUTPUT_DIR.
# Depends on: block 2 (OUTPUT_DIR), block 4 (EFFECTIVE_MODEL), block 7 (trained_model).
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(EFFECTIVE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Save the resulting trained_model + tokenizer
trained_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved to", OUTPUT_DIR)
