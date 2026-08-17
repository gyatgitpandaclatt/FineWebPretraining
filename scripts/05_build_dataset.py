# Block 5 of 10 — wraps token blocks into HF Datasets + builds the LM data collator.
# Depends on: block 4 (EFFECTIVE_MODEL, train_blocks, eval_blocks).
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, AutoTokenizer

# Re-load tokenizer (imports don't carry between blocks)
tokenizer = AutoTokenizer.from_pretrained(EFFECTIVE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

train_ds = Dataset.from_list(train_blocks)
eval_ds  = Dataset.from_list(eval_blocks)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)  # causal LM
print(f"train_ds: {len(train_ds)} rows   eval_ds: {len(eval_ds)} rows")
