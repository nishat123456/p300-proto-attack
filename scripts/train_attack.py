"""Train one prototype-guided attack generator for one Comp III subject.

The training math lives in ``p3a.attacks.train.train_attack``; this script only
parses arguments, enforces the local smoke constraints, and prints the result.

Local smoke: ``--smoke`` trains on the first 4 frozen train trials and 3 frozen
val trials for one epoch on CPU, using the real loss and data flow. It never
loads the test split, so a smoke run is safe on a laptop with only
``Subject_A_Train.mat``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from p3a.attacks.train import (  # noqa: E402
    BANKS,
    GENERATORS,
    METHODS,
    train_attack,
)

DEFAULT_SMOKE_SURROGATE = REPO / "results/smoke_prompt4/A_eegnet_s0.pt"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(REPO / "data/raw/comp3"))
    p.add_argument("--labels-dir", default=None)
    p.add_argument("--subject", required=True, choices=("A", "B"))
    p.add_argument("--surrogate", required=True,
                   help="EEGNet Speller checkpoint (raw EEG in, symbols out)")
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--generator", required=True, choices=GENERATORS)
    p.add_argument("--bank", default="cross_subject", choices=BANKS)
    p.add_argument("--lambda-p", dest="lambda_p", type=float, default=1.0)
    p.add_argument("--lambda-s", dest="lambda_s", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(REPO / "results/attacks"))
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch, 4 train / 3 val trials, CPU (local)")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.smoke:
        args.epochs = 1
        if args.device == "auto":
            args.device = "cpu"
    result = train_attack(
        data_root=args.data_root, labels_dir=args.labels_dir, subject=args.subject,
        surrogate=args.surrogate, method=args.method, generator=args.generator,
        bank=args.bank, seed=args.seed, out_dir=args.out_dir,
        lambda_p=args.lambda_p, lambda_s=args.lambda_s, epochs=args.epochs,
        batch=args.batch, lr=args.lr, device=args.device, smoke=args.smoke,
    )
    printable = {k: v for k, v in result.items() if k != "history"}
    print(json.dumps(printable, indent=2))
    return result


if __name__ == "__main__":
    main()
