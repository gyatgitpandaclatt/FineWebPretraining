# Block 4 of 10 — tokenizer + block_generator; materializes train/eval token blocks.
# Depends on: block 2 (MODEL_NAME, DATASET_CONFIG, MIN_EDU_SCORE, BLOCK_SIZE, N_TRAIN_BLOCKS, N_EVAL_BLOCKS, RUN_STATE_DIR).
# Defines EFFECTIVE_MODEL, train_blocks, eval_blocks used by every later block.
import itertools
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def block_generator(stream, n_blocks, min_score, block_size):
    """Tokenize filtered docs and yield non-overlapping token blocks of exactly block_size."""
    token_buffer = []
    blocks_yielded = 0
    for doc in stream:
        if doc.get("score", 0) < min_score:
            continue
        ids = tokenizer(doc["text"], add_special_tokens=False)["input_ids"]
        token_buffer.extend(ids)
        while len(token_buffer) >= block_size and blocks_yielded < n_blocks:
            yield {"input_ids": token_buffer[:block_size]}
            token_buffer = token_buffer[block_size:]
            blocks_yielded += 1
        if blocks_yielded >= n_blocks:
            break

# Materialize train + eval blocks from a fresh stream so they don't overlap.
_stream = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name=DATASET_CONFIG,
    split="train",
    streaming=True,
)
_gen = block_generator(_stream, N_TRAIN_BLOCKS + N_EVAL_BLOCKS, MIN_EDU_SCORE, BLOCK_SIZE)

all_blocks   = list(_gen)
train_blocks = all_blocks[:N_TRAIN_BLOCKS]
eval_blocks  = all_blocks[N_TRAIN_BLOCKS:]
EFFECTIVE_MODEL = MODEL_NAME
print(f"Model:        {EFFECTIVE_MODEL}")
print(f"train blocks: {len(train_blocks)}   eval blocks: {len(eval_blocks)}")
print(f"total training tokens: {len(train_blocks) * BLOCK_SIZE:,}")

(Path(RUN_STATE_DIR) / "dataset_stats.json").write_text(json.dumps({
    "effective_model": EFFECTIVE_MODEL,
    "train_blocks": len(train_blocks),
    "eval_blocks": len(eval_blocks),
    "total_training_tokens": len(train_blocks) * BLOCK_SIZE,
}, indent=2))
