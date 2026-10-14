# Runs the pipeline blocks (01, 02, 04-10) in order in one shared namespace, the way the
# Zerve canvas does, so `python scripts/run_pipeline.py` reproduces a full run and feeds
# ./run_state for dashboard/app.py.
#
# Usage (from the repo root):
#   python scripts/run_pipeline.py                         # full run
#   python scripts/run_pipeline.py --dashboard             # also launch the Streamlit dashboard
#   python scripts/run_pipeline.py --set MAX_STEPS=20 --set N_TRAIN_BLOCKS=200   # quick smoke run
#
# --set overrides are applied right after block 02 defines the config, and config.json is
# rewritten so the dashboard sidebar shows the values actually used.
import argparse
import ast
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
# Block 03 is standalone exploration that later blocks don't depend on, so it's skipped.
PIPELINE_BLOCKS = [
    "01_setup.py",
    "02_config.py",
    "04_tokenize_blocks.py",
    "05_build_dataset.py",
    "06_eval_perplexity_before.py",
    "07_train.py",
    "08_eval_perplexity_after.py",
    "09_generate_sample.py",
    "10_save_model.py",
]
CONFIG_BLOCK = "02_config.py"


def parse_override(text):
    """Parse KEY=VALUE, evaluating VALUE as a Python literal when possible (else a string)."""
    key, sep, raw = text.partition("=")
    if not sep or not key.strip():
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw
    return key.strip(), value


def _update_status(ns, **fields):
    """Merge fields into run_state/status.json, once block 02 has defined RUN_STATE_DIR."""
    run_state = ns.get("RUN_STATE_DIR")
    if run_state is None:
        return
    path = Path(run_state) / "status.json"
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        data = {}
    data.update(fields, updated_at=time.time())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _apply_overrides(ns, overrides):
    unknown = [k for k in overrides if k not in ns]
    if unknown:
        raise KeyError(f"--set keys not defined by {CONFIG_BLOCK}: {', '.join(unknown)}")
    ns.update(overrides)
    config_path = Path(ns["RUN_STATE_DIR"]) / "config.json"
    config = json.loads(config_path.read_text())
    config.update({k: v for k, v in overrides.items() if k in config})
    config_path.write_text(json.dumps(config, indent=2))
    for k, v in overrides.items():
        print(f"override: {k} = {v!r}")


def run_blocks(block_paths, overrides=None):
    """Exec each block in a single shared namespace. Returns the namespace."""
    ns = {"__name__": "__fineweb_pipeline__"}
    for path in block_paths:
        path = Path(path)
        name = path.stem
        print(f"\n===== {name} =====", flush=True)
        _update_status(ns, stage="running", block=name)
        try:
            ns["__file__"] = str(path)
            exec(compile(path.read_text(), str(path), "exec"), ns)
            if path.name == CONFIG_BLOCK and overrides:
                _apply_overrides(ns, overrides)
        except BaseException as exc:
            _update_status(ns, stage="failed", block=name, error=f"{type(exc).__name__}: {exc}")
            raise
    return ns


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the FineWeb-Edu pipeline blocks in one session, feeding ./run_state for the dashboard.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], type=parse_override,
                        metavar="KEY=VALUE", help="override a block 02 config constant (repeatable)")
    parser.add_argument("--dashboard", action="store_true",
                        help="also launch `streamlit run dashboard/app.py` for the duration of the run")
    args = parser.parse_args(argv)
    os.chdir(REPO_ROOT)  # blocks write ./run_state and ./fineweb-edu-run relative to the cwd

    dashboard = None
    if args.dashboard:
        dashboard = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(REPO_ROOT / "dashboard" / "app.py")],
            cwd=REPO_ROOT,
        )

    try:
        run_blocks([SCRIPTS_DIR / b for b in PIPELINE_BLOCKS], dict(args.overrides))
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if dashboard is not None:
            print("\nRun finished; dashboard still serving. Ctrl+C to stop it.")
            try:
                dashboard.wait()
            except KeyboardInterrupt:
                dashboard.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
