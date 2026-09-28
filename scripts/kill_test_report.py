"""Build the kill-test report from a downloaded Kaggle output (Prompt 12).

Thin CLI over ``p3a.report.kill_test``: point it at the directory holding the
downloaded ``out/kill_test/`` tree and it writes ``results/kill_test_report.md``
with the PNG plots next to it. Nothing is trained and no attack is re-run.

Typical use after ``bash scripts/kaggle.sh output kill_test``:

    .venv/bin/python scripts/kill_test_report.py \
        --output-root kaggle/out/kill_test/out/kill_test

Use ``--out-dir`` to write the report somewhere else; the file name stays
``kill_test_report.md`` and the plots land beside it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from p3a.report.kill_test import build_report  # noqa: E402
from p3a.utils.repro import environment, set_seed, write_config  # noqa: E402

DEFAULT_OUTPUT_ROOT = REPO / "kaggle/out/kill_test/out/kill_test"
DEFAULT_OUT_DIR = REPO / "results"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT),
                   help="directory holding decoders/, attacks_eval/ and <subject>/ CSVs")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                   help="directory for kill_test_report.md and the PNG plots "
                        "(default: results/)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)
    result = build_report(args.output_root, args.out_dir)
    config = {
        "output_root": str(args.output_root),
        "out_dir": str(args.out_dir),
        "seed": int(args.seed),
        "verdict": result["verdict"],
        "report": result["report"],
        "plots": result["plots"],
        "n_attack_rows": result["n_attack_rows"],
        "n_detector_rows": result["n_detector_rows"],
        "provenance": environment(args.seed),
    }
    write_config(args.out_dir, "kill_test_report.config.json", config)
    print(json.dumps({k: v for k, v in config.items() if k != "provenance"}, indent=2))
    return result


if __name__ == "__main__":
    main()
