# Block 10 of 10 — saves the trained model + tokenizer to OUTPUT_DIR.
# Depends on: block 2 (OUTPUT_DIR, RUN_STATE_DIR), block 4 (EFFECTIVE_MODEL), block 7 (trained_model).
import json
import time
from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(EFFECTIVE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Save the resulting trained_model + tokenizer
trained_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved to", OUTPUT_DIR)

_status_path = Path(RUN_STATE_DIR) / "status.json"
_status_data = json.loads(_status_path.read_text()) if _status_path.exists() else {}
_status_data.update({"stage": "saved", "output_dir": OUTPUT_DIR, "updated_at": time.time()})
_status_path.write_text(json.dumps(_status_data, indent=2))
