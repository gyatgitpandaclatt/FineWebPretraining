# Block 3 of 10 — exploratory: education-score histogram over a small sample.
# Independent of blocks 1-2 (loads its own stream); not depended on by later blocks.
import itertools
from collections import Counter
from datasets import load_dataset

# Look at the education-score distribution over a small sample.
# `score` is the classifier's 0-5 educational-quality rating; `int_score` is its rounded version.
# Load a fresh stream locally — IterableDatasets are not serializable between blocks.
_dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
)

sample = list(itertools.islice(_dataset, 500))
score_hist = Counter(round(r["score"]) for r in sample)
for s in sorted(score_hist):
    bar = "X" * score_hist[s]
    print(f"score = {s}: {score_hist[s]:>3}  {bar}")

print("\nExample of a high-scoring doc:")
best = max(sample, key=lambda r: r["score"])
print(f"  score={best['score']:.2f}  url={best.get('url')}")
print(" ", best["text"][:400])
