# Block 8 of 10 — measures perplexity after training and reports the % change.
# Depends on: block 6 (ppl_before, eval_blocks), block 7 (trained_model).
import math
import torch

_device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def perplexity(mdl, blocks, batch_size=8):
    mdl.eval()
    losses = []
    for i in range(0, len(blocks), batch_size):
        batch = blocks[i:i + batch_size]
        ids = torch.tensor([b["input_ids"] for b in batch], device=_device)
        out = mdl(input_ids=ids, labels=ids)
        losses.append(out.loss.item())
    return math.exp(sum(losses) / len(losses))

ppl_after = perplexity(trained_model, eval_blocks)
print(f"Perplexity BEFORE: {ppl_before:.2f}")
print(f"Perplexity AFTER:  {ppl_after:.2f}")
delta = 100 * (ppl_before - ppl_after) / ppl_before
print(f"Change: {delta:+.1f}%  (negative = model improved on this data)")
