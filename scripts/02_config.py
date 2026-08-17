# Block 2 of 10 — run config. Defines constants read by every later block.
# Also (re)initializes ./run_state, the on-disk state that dashboard/app.py polls,
# so each fresh run starts with a clean dashboard.
import json
import time
from pathlib import Path

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
RUN_STATE_DIR    = "./run_state"   # read by dashboard/app.py; written by blocks 02/04/06/07/08/09/10

_run_state = Path(RUN_STATE_DIR)
_run_state.mkdir(parents=True, exist_ok=True)
for _stale in ("metrics.jsonl", "status.json", "dataset_stats.json", "perplexity.json", "sample.json"):
    (_run_state / _stale).unlink(missing_ok=True)

(_run_state / "config.json").write_text(json.dumps({
    "MODEL_NAME": MODEL_NAME,
    "DATASET_CONFIG": DATASET_CONFIG,
    "MIN_EDU_SCORE": MIN_EDU_SCORE,
    "BLOCK_SIZE": BLOCK_SIZE,
    "N_TRAIN_BLOCKS": N_TRAIN_BLOCKS,
    "N_EVAL_BLOCKS": N_EVAL_BLOCKS,
    "MAX_STEPS": MAX_STEPS,
    "BATCH_SIZE": BATCH_SIZE,
    "GRAD_ACCUM": GRAD_ACCUM,
    "LR": LR,
    "OUTPUT_DIR": OUTPUT_DIR,
    "started_at": time.time(),
}, indent=2))
