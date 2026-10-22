# Live dashboard for the FineWeb-Edu pretraining pipeline in ../scripts.
#
# Run from the repo root (so the "run_state" path below matches where the
# scripts write it):
#   streamlit run dashboard/app.py
#
# It reads the JSON/JSONL files scripts/02, 04, 06, 07, 08, 09, 10 write to
# ./run_state while a pipeline session (see README for run order) is in
# progress or has finished, and auto-refreshes every 2 seconds.
import json
from pathlib import Path

import pandas as pd
import streamlit as st

RUN_STATE_DIR = Path("run_state")

STAGE_LABELS = {
    "running": "Running",
    "failed": "Failed",
    "training": "Training",
    "training_complete": "Training complete",
    "saved": "Model saved",
}


def _read_json(name):
    path = RUN_STATE_DIR / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _read_metrics():
    path = RUN_STATE_DIR / "metrics.jsonl"
    if not path.exists():
        return pd.DataFrame(columns=["step", "loss", "lr"])
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return pd.DataFrame(rows)


st.set_page_config(page_title="FineWeb-Edu Pretraining", layout="wide")
st.title("FineWeb-Edu Pretraining Dashboard")

config = _read_json("config.json")

if config is None:
    st.info("No run detected yet. Start one with `python scripts/run_pipeline.py` (see README).")
    st.stop()

with st.sidebar:
    st.header("Run config")
    st.json(config)


@st.fragment(run_every=2)
def status_section():
    status = _read_json("status.json")
    st.subheader("Status")
    if status is None:
        st.caption("Waiting for the run to start...")
        return
    stage = status.get("stage", "unknown")
    step = status.get("step")
    max_steps = status.get("max_steps")
    label = STAGE_LABELS.get(stage, stage)
    if stage in ("running", "failed") and status.get("block"):
        label += f" — {status['block']}"
    st.write(f"**{label}**")
    if stage == "failed" and status.get("error"):
        st.error(status["error"])
    if step is not None and max_steps:
        st.progress(min(step / max_steps, 1.0), text=f"step {step}/{max_steps}")


def dataset_section():
    stats = _read_json("dataset_stats.json")
    st.subheader("Dataset")
    if stats is None:
        st.caption("Not tokenized yet (block 04).")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Effective model", stats["effective_model"])
    c2.metric("Train blocks", stats["train_blocks"])
    c3.metric("Eval blocks", stats["eval_blocks"])
    c4.metric("Total training tokens", f"{stats['total_training_tokens']:,}")


@st.fragment(run_every=2)
def loss_section():
    metrics = _read_metrics()
    st.subheader("Training loss")
    if metrics.empty:
        st.caption("No training steps logged yet (block 07).")
        return
    col1, col2 = st.columns([3, 1])
    with col1:
        st.line_chart(metrics.set_index("step")[["loss"]])
    with col2:
        st.metric("Latest loss", f"{metrics['loss'].iloc[-1]:.4f}")
        st.metric("Latest LR", f"{metrics['lr'].iloc[-1]:.2e}")


@st.fragment(run_every=2)
def perplexity_section():
    ppl = _read_json("perplexity.json")
    st.subheader("Perplexity")
    if not ppl or "ppl_before" not in ppl:
        st.caption("Not evaluated yet (block 06).")
        return
    c1, c2 = st.columns(2)
    c1.metric("Before training", f"{ppl['ppl_before']:.2f}")
    if "ppl_after" in ppl:
        c2.metric(
            "After training",
            f"{ppl['ppl_after']:.2f}",
            delta=f"{ppl['delta_pct']:+.1f}%",
            delta_color="inverse",  # lower perplexity is better, so a negative delta should read as good
        )
    else:
        c2.metric("After training", "pending (block 08)")


@st.fragment(run_every=2)
def sample_section():
    sample = _read_json("sample.json")
    st.subheader("Sample generation")
    if sample is None:
        st.caption("No sample generated yet (block 09).")
        return
    st.text_area("Prompt", sample["prompt"], height=68, disabled=True)
    st.text_area("Generated", sample["generated"], height=160, disabled=True)


status_section()
dataset_section()
loss_section()
perplexity_section()
sample_section()
