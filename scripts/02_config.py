# Block 2 of 10 — run config. Defines constants read by every later block.
# ---- CONFIG ----
MODEL_NAME       = "Qwen/Qwen2.5-0.5B"
DATASET_CONFIG   = "sample-10BT"  # streamed, so this doesn't download the whole slice
MIN_EDU_SCORE    = 3.0             # keep only higher-quality educational text (score is ~0–5)
BLOCK_SIZE       = 512             # sequence length in tokens
N_TRAIN_BLOCKS   = 2000            # number of training blocks to materialize
N_EVAL_BLOCKS    = 200             # held-out blocks for perplexity
MAX_STEPS        = 200             # training steps (raise for real runs)
BATCH_SIZE       = 4
GRAD_ACCUM       = 4              # effective batch = BATCH_SIZE * GRAD_ACCUM
LR               = 5e-5
OUTPUT_DIR       = "./fineweb-edu-run"
