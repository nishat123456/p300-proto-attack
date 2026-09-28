"""Train Tiny Noise raw templates for one Comp III subject (Prompt 9).

One universal raw template is trained per epsilon multiplier on the frozen
TRAIN trials, then saved with a config sidecar. The training math lives in
``p3a.attacks.baselines.TinyNoiseTemplate`` and ``train_tiny_noise``; this
script only parses arguments and prints the result.

Local smoke: ``--smoke`` trains on the first few frozen train trials for one
epoch on CPU using the real loss and placement. It never loads the test split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from p3a.attacks.baselines import train_tiny_noise  # noqa: E402

DEFAULT_SMOKE_SURROGATE = REPO / "results/smoke_prompt4/A_eegnet_s0.pt"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(REPO / "data/raw/comp3"))
    p.add_argument("--labels-dir", default=None)
    p.add_argument("--subject", required=True, choices=("A", "B"))
    p.add_argument("--surrogate", required=True,
                   help="EEGNet Speller checkpoint (raw EEG in, symbols out)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(REPO / "results/tiny_noise"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--template-len", dest="template_len", type=int, default=192)
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch, few train trials, CPU (local)")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.smoke:
        args.epochs = 1
        if args.device == "auto":
            args.device = "cpu"
    result = train_tiny_noise(
        data_root=args.data_root, labels_dir=args.labels_dir, subject=args.subject,
        surrogate=args.surrogate, seed=args.seed, out_dir=args.out_dir,
        epochs=args.epochs, lr=args.lr, device=args.device,
        template_len=args.template_len, smoke=args.smoke,
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
