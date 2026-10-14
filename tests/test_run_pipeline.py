import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "run_pipeline", Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"
)
run_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_pipeline)


def _write_blocks(tmp_path, run_state, extra_body="y = X * 2\n"):
    config = tmp_path / "02_config.py"
    config.write_text(
        "import json\nfrom pathlib import Path\n"
        f"RUN_STATE_DIR = {str(run_state)!r}\nX = 1\nMAX_STEPS = 200\n"
        "Path(RUN_STATE_DIR).mkdir(parents=True, exist_ok=True)\n"
        "(Path(RUN_STATE_DIR) / 'config.json').write_text(json.dumps({'X': X, 'MAX_STEPS': MAX_STEPS}))\n"
    )
    later = tmp_path / "04_later.py"
    later.write_text(extra_body)
    return [tmp_path / "01_setup.py", config, later]


def test_parse_override():
    assert run_pipeline.parse_override("MAX_STEPS=20") == ("MAX_STEPS", 20)
    assert run_pipeline.parse_override("LR=5e-5") == ("LR", 5e-5)
    assert run_pipeline.parse_override("MODEL_NAME=EleutherAI/pythia-70m") == ("MODEL_NAME", "EleutherAI/pythia-70m")


def test_blocks_share_namespace_and_overrides_apply(tmp_path):
    run_state = tmp_path / "run_state"
    blocks = _write_blocks(tmp_path, run_state)
    blocks[0].write_text("setup_done = True\n")
    ns = run_pipeline.run_blocks(blocks, {"X": 5, "MAX_STEPS": 20})
    assert ns["setup_done"] and ns["y"] == 10
    assert json.loads((run_state / "config.json").read_text()) == {"X": 5, "MAX_STEPS": 20}
    status = json.loads((run_state / "status.json").read_text())
    assert status["stage"] == "running" and status["block"] == "04_later"


def test_unknown_override_rejected(tmp_path):
    blocks = _write_blocks(tmp_path, tmp_path / "run_state")
    blocks[0].write_text("")
    with pytest.raises(KeyError):
        run_pipeline.run_blocks(blocks, {"TYPO": 1})


def test_failure_recorded_in_status(tmp_path):
    run_state = tmp_path / "run_state"
    blocks = _write_blocks(tmp_path, run_state, extra_body="raise RuntimeError('boom')\n")
    blocks[0].write_text("")
    with pytest.raises(RuntimeError):
        run_pipeline.run_blocks(blocks)
    status = json.loads((run_state / "status.json").read_text())
    assert status["stage"] == "failed"
    assert status["block"] == "04_later"
    assert status["error"] == "RuntimeError: boom"
