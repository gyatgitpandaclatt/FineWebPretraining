# Block 7 of 10 — the training loop (AdamW + cosine schedule, gradient accumulation).
# Depends on: block 2 (LR, MAX_STEPS, GRAD_ACCUM, BATCH_SIZE, RUN_STATE_DIR), block 4 (EFFECTIVE_MODEL, train_blocks).
# Defines trained_model used by later blocks. Appends one line per step to
# run_state/metrics.jsonl and updates run_state/status.json — dashboard/app.py polls both.
import json
import time
from pathlib import Path
import torch
import random
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM

_device = "cuda" if torch.cuda.is_available() else "cpu"

_metrics_path = Path(RUN_STATE_DIR) / "metrics.jsonl"
_status_path = Path(RUN_STATE_DIR) / "status.json"

def _write_status(stage, **extra):
    _status_path.write_text(json.dumps({"stage": stage, "updated_at": time.time(), **extra}, indent=2))

_write_status("training", step=0, max_steps=MAX_STEPS)

# causal_model is deserialized as GPTNeoXModel (base, no LM head) by Zerve.
# Load fresh from pretrained as ForCausalLM — same weights as upstream block used.
_model = AutoModelForCausalLM.from_pretrained(EFFECTIVE_MODEL).to(_device)

optimizer = AdamW(_model.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=max(1, MAX_STEPS - int(0.03 * MAX_STEPS)))

_model.train()
optimizer.zero_grad()
_step = 0
_accum_loss = 0.0

_indices = list(range(len(train_blocks)))
random.shuffle(_indices)
_idx_pos = 0

while _step < MAX_STEPS:
    _step_loss = 0.0
    for _micro in range(GRAD_ACCUM):
        _batch_idx = []
        for _ in range(BATCH_SIZE):
            if _idx_pos >= len(_indices):
                random.shuffle(_indices)
                _idx_pos = 0
            _batch_idx.append(_indices[_idx_pos])
            _idx_pos += 1

        _ids = torch.tensor(
            [train_blocks[i]["input_ids"] for i in _batch_idx], device=_device
        )
        _out = _model(input_ids=_ids, labels=_ids)
        _loss = _out.loss / GRAD_ACCUM
        _loss.backward()
        _accum_loss += _loss.item()
        _step_loss += _out.loss.item()

    torch.nn.utils.clip_grad_norm_(_model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    _step += 1

    with open(_metrics_path, "a") as _mf:
        _mf.write(json.dumps({
            "step": _step,
            "loss": _step_loss / GRAD_ACCUM,
            "lr": scheduler.get_last_lr()[0],
            "t": time.time(),
        }) + "\n")
    _write_status("training", step=_step, max_steps=MAX_STEPS)

    if _step % 20 == 0 or _step == MAX_STEPS:
        print(f"  step {_step:>4}/{MAX_STEPS}  loss={_accum_loss / 20:.4f}")
        _accum_loss = 0.0

print("Training complete.")
trained_model = _model
_write_status("training_complete", step=_step, max_steps=MAX_STEPS)
