# FineWebPretraining

Scripts for pretraining/fine-tuning a small causal LM on a quality-filtered slice of
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu): stream the
dataset, keep only documents above an education-score threshold, pack them into
fixed-size token blocks, train, and measure perplexity before/after.

## About `scripts/`

These files were exported from a block-based notebook canvas (Zerve), **not** written as
independent standalone scripts. Each block shares a single in-memory session with the
blocks before it — variables like `EFFECTIVE_MODEL`, `train_blocks`, `eval_blocks`,
`causal_model`, `ppl_before`, and `trained_model` are defined in one block and read by
later ones without being redefined there.

Run order:

| File | Depends on | Defines |
|---|---|---|
| `01_setup.py` | — | `device` |
| `02_config.py` | — | run config constants |
| `03_explore_fineweb_scores.py` | — (standalone exploration, not required by later blocks) | — |
| `04_tokenize_blocks.py` | 02 | `EFFECTIVE_MODEL`, `train_blocks`, `eval_blocks` |
| `05_build_dataset.py` | 04 | `train_ds`, `eval_ds`, `collator` |
| `06_eval_perplexity_before.py` | 04 | `causal_model`, `ppl_before` |
| `07_train.py` | 02, 04 | `trained_model` |
| `08_eval_perplexity_after.py` | 06, 07 | `ppl_after` |
| `09_generate_sample.py` | 04, 07 | — |
| `10_save_model.py` | 02, 04, 07 | — |

To reproduce this as a normal Python session, run blocks 01, 02, 04, 05, 06, 07, 08, 09, 10
in order in the **same interpreter** (e.g. paste them into one REPL/Jupyter session, or
`exec()` each file's contents in sequence) — running any file with `python scripts/07_train.py`
on its own will `NameError` on names like `EFFECTIVE_MODEL` that a prior block defines.
Block 03 is independent and can be run on its own.

## Setup

```
pip install -e .
```

`transformers`, `torch`, and `datasets` are declared in `pyproject.toml`. Note
`04_tokenize_blocks.py` falls back from `Qwen/Qwen2.5-0.5B` to `EleutherAI/pythia-160m`
when `transformers<4.43` is installed.
