"""Download, validate, and emit frozen split manifests for the two BNCI sets."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from p3a.data import splits  # noqa: E402
from p3a.data.bnci_splits import (  # noqa: E402
    attack_targets,
    chronological_70_15_15,
    official_train_test,
)
from p3a.data.loaders.bnci2014_008 import load_subject as load_008  # noqa: E402
from p3a.data.loaders.bnci2015_003 import load_subject as load_003  # noqa: E402
from p3a.utils.repro import environment  # noqa: E402

BASE = "https://bnci-horizon-2020.eu/database/data-sets"
SUBJECTS_008 = [f"A{i:02d}" for i in range(1, 9)]
SUBJECTS_003 = [f"s{i}" for i in range(1, 11)]


def _download(url: str, path: Path):
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(path)


def prepare(root: Path, out: Path, seed: int = 20260929,
            download: bool = True) -> dict:
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    audit = {"seed": seed, "datasets": {}, "provenance": environment(seed)}

    records = []
    for offset, subject in enumerate(SUBJECTS_008):
        path = root / "bnci2014_008" / f"{subject}.mat"
        if download:
            _download(f"{BASE}/008-2014/{subject}.mat", path)
        trials = load_008(path.parent, subject)
        split = chronological_70_15_15(trials)
        split_path = out / "splits" / f"bnci2014_008_{subject}.json"
        split_hash = splits.write_once(split_path, split)
        test_ids = set(split["test"])
        targets = attack_targets([trial for trial in trials if trial.trial_id in test_ids],
                                 seed + offset)
        target_hash = splits.write_once(
            out / "splits" / f"bnci2014_008_{subject}_attack_targets.json", targets
        )
        records.append({"subject": subject, "trials": len(trials),
                        "flashes_per_trial": trials[0].n_flashes,
                        "split_counts": {key: len(value) for key, value in split.items()},
                        "split_hash": split_hash, "target_hash": target_hash})
    audit["datasets"]["bnci2014_008"] = records

    records = []
    for offset, subject in enumerate(SUBJECTS_003):
        path = root / "bnci2015_003" / f"{subject}.mat"
        if download:
            _download(f"{BASE}/003-2015/{subject}.mat", path)
        train = load_003(path.parent, subject, "train")
        test = load_003(path.parent, subject, "test")
        split = official_train_test(train, test)
        split_path = out / "splits" / f"bnci2015_003_{subject}.json"
        split_hash = splits.write_once(split_path, split)
        targets = attack_targets(test, seed + 100 + offset)
        target_hash = splits.write_once(
            out / "splits" / f"bnci2015_003_{subject}_attack_targets.json", targets
        )
        records.append({"subject": subject, "train_trials": len(train),
                        "test_trials": len(test),
                        "flashes_per_trial": train[0].n_flashes,
                        "split_counts": {key: len(value) for key, value in split.items()},
                        "split_hash": split_hash, "target_hash": target_hash})
    audit["datasets"]["bnci2015_003"] = records
    (out / "bnci_data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)
    result = prepare(Path(args.data_root), Path(args.out_dir), args.seed,
                     download=not args.no_download)
    print(json.dumps({name: len(rows) for name, rows in result["datasets"].items()},
                     indent=2))


if __name__ == "__main__":
    main()
