# Block 6 of 10 — loads the base model and measures perplexity before training.
# Depends on: block 4 (EFFECTIVE_MODEL, eval_blocks).
# Defines causal_model, ppl_before used by later blocks.
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

_device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def perplexity(mdl, blocks, batch_size=8):
    mdl.eval()
    losses = []
    for i in range(0, len(blocks), batch_size):
        batch = blocks[i:i + batch_size]
        ids = torch.tensor([b["input_ids"] for b in batch], device=_device)
        # AutoModelForCausalLM returns CausalLMOutputWithPast which has .loss when labels provided
        out = mdl(input_ids=ids, labels=ids)
        losses.append(out.loss.item())
    return math.exp(sum(losses) / len(losses))

# Load with AutoModelForCausalLM — gives GPTNeoXForCausalLM (with LM head, supports labels=)
causal_model = AutoModelForCausalLM.from_pretrained(EFFECTIVE_MODEL).to(_device)
ppl_before = perplexity(causal_model, eval_blocks)
print(f"Device:            {_device}")
print(f"Model type:        {type(causal_model).__name__}")
print(f"Perplexity BEFORE: {ppl_before:.2f}")
