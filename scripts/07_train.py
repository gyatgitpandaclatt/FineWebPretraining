# Block 7 of 10 — the training loop (AdamW + cosine schedule, gradient accumulation).
# Depends on: block 2 (LR, MAX_STEPS, GRAD_ACCUM, BATCH_SIZE), block 4 (EFFECTIVE_MODEL, train_blocks).
# Defines trained_model used by later blocks.
import torch
import random
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM

_device = "cuda" if torch.cuda.is_available() else "cpu"

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

    torch.nn.utils.clip_grad_norm_(_model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    _step += 1

    if _step % 20 == 0 or _step == MAX_STEPS:
        print(f"  step {_step:>4}/{MAX_STEPS}  loss={_accum_loss / 20:.4f}")
        _accum_loss = 0.0

print("Training complete.")
trained_model = _model
