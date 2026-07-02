---
description: Run a calibration script for a scoring criterion
argument-hint: [clarity|completeness|relevance|retrieval_relevance|threshold]
---

Run the calibration script for criterion `$ARGUMENTS`.

Map `$ARGUMENTS` to the script under `scripts/`:
- `clarity` → `scripts/calibrate_clarity.py`
- `completeness` → `scripts/calibrate_completeness.py`
- `relevance` → `scripts/calibrate_relevance.py`
- `retrieval_relevance` → `scripts/calibrate_retrieval_relevance.py`
- `threshold` → `scripts/calibrate_threshold.py`

If `$ARGUMENTS` doesn't match one of these, list the valid options and stop.

Otherwise run `python scripts/calibrate_<criterion>.py` from the project root and report the recommended bounds it prints.
